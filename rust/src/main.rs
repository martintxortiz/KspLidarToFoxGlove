//! Stream KSP LiDAR points to Foxglove as an accumulating, height-coloured 3D
//! scan, plus a box marker that tracks the vehicle pose. Rust port of the
//! Python bridge -- the hot per-point loop runs natively for speed.

use std::collections::HashSet;

use foxglove::messages::packed_element_field::NumericType;
use foxglove::messages::{
    Color, CubePrimitive, PackedElementField, PointCloud, Pose, Quaternion, SceneEntity,
    SceneUpdate, Timestamp, Vector3,
};
use foxglove::LazyChannel;

use krpc_client::services::li_dar::LiDAR;
use krpc_client::services::space_center::SpaceCenter;
use krpc_client::Client;

// --- settings ---------------------------------------------------------------
// Where KSP/kRPC is running. "127.0.0.1" = same machine. Set to the game PC's
// LAN IP (e.g. "192.168.1.166") to run the bridge from another machine.
const KRPC_HOST: &str = "192.168.1.166";
const TAG: &str = "laser";
const TICK_SPACING: f32 = 0.0;
const FRAME: &str = "world";
const PUBLISH_HZ: f64 = 10.0;
const FLIP_Z: bool = true;

const VOXEL: f64 = 0.1;
const MAX_POINTS: usize = 500_000;
const MIN_RANGE: f64 = 0.0;
const MAX_RANGE: f64 = 0.0;

// The cloud arrives in KSP's world axes, tilted/twisted relative to the local
// horizon, so planes look rotated vs the box. AUTO_ALIGN fixes this by itself: it
// fits the ground plane in the scan to find true "up" (levels it), then snaps the
// dominant wall direction to the grid (removes the yaw twist). No hand tuning.
// The CLOUD_*_DEG values are an optional extra nudge applied on top (usually 0).
const AUTO_ALIGN: bool = true;
const AUTO_YAW: bool = true; // also snap walls to the grid, not just level
const CALIB_MIN_POINTS: usize = 15000; // points to gather before computing the alignment
const CALIB_MIN_RANGE: f64 = 1.0; // metres of vertical spread needed before calibrating
const CLOUD_YAW_DEG: f64 = 0.0; // optional extra rotation about vertical (z)
const CLOUD_PITCH_DEG: f64 = 0.0; // optional extra fore/aft tilt
const CLOUD_ROLL_DEG: f64 = 0.0; // optional extra side/side tilt

const BOX_SIZE: [f64; 3] = [3.0, 3.0, 1.5];
const BOX_TRACK_ORIENTATION: bool = true;
const YAW_OFFSET_DEG: f64 = 0.0;
const BOX_SMOOTHING: f64 = 0.15;

const ROTOR_RPM: f64 = 120.0;
const SWEEP_REVS: f64 = 0.5;
const BEND_FIELD: &str = "Bend Y";
const BEND_LIMIT: f64 = 0.0; // 0 = read "Max Bend Y" from the part

const COLOR_LO: f64 = -2.0;
const COLOR_HI: f64 = 20.0;

const LIDAR_MODULES: [&str; 2] = ["LiDARModule", "LaserDistModule"];

static CLOUD: LazyChannel<PointCloud> = LazyChannel::new("/ksp/lidar/points");
static VEHICLE: LazyChannel<SceneUpdate> = LazyChannel::new("/ksp/vehicle");

fn triangle(t: f64, half_period: f64, lo: f64, hi: f64) -> f64 {
    let phase = (t % (2.0 * half_period)) / half_period; // 0..2
    let frac = if phase <= 1.0 { phase } else { 2.0 - phase };
    lo + (hi - lo) * frac
}

/// kRPC heading/pitch/roll -> quaternion (x, y, z, w) in a z-up cloud frame.
fn euler_to_quat(heading_deg: f64, pitch_deg: f64, roll_deg: f64) -> [f64; 4] {
    let psi = (YAW_OFFSET_DEG - heading_deg).to_radians();
    let theta = pitch_deg.to_radians();
    let phi = roll_deg.to_radians();
    let (cy, sy) = ((psi / 2.0).cos(), (psi / 2.0).sin());
    let (cp, sp) = ((theta / 2.0).cos(), (theta / 2.0).sin());
    let (cr, sr) = ((phi / 2.0).cos(), (phi / 2.0).sin());
    [
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ]
}

/// Fixed 3x3 cloud-alignment matrix R = Rz(yaw) * Ry(pitch) * Rx(roll), degrees.
fn cloud_rotation() -> [[f64; 3]; 3] {
    let (y, p, r) = (
        CLOUD_YAW_DEG.to_radians(),
        CLOUD_PITCH_DEG.to_radians(),
        CLOUD_ROLL_DEG.to_radians(),
    );
    let (cy, sy) = (y.cos(), y.sin());
    let (cp, sp) = (p.cos(), p.sin());
    let (cr, sr) = (r.cos(), r.sin());
    [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]
}

fn rotate3(m: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

fn mat_mul(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut r = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    r
}

/// Rotation matrix that turns unit vector `u` onto unit vector `v` (Rodrigues).
fn rotation_between(u: [f64; 3], v: [f64; 3]) -> [[f64; 3]; 3] {
    let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let d = (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]).clamp(-1.0, 1.0);
    if d > 0.999999 {
        return identity;
    }
    if d < -0.999999 {
        return [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]];
    }
    let ax = [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ];
    let s = (ax[0] * ax[0] + ax[1] * ax[1] + ax[2] * ax[2]).sqrt();
    let (x, y, z) = (ax[0] / s, ax[1] / s, ax[2] / s);
    let c = d;
    let k = 1.0 - c;
    [
        [c + x * x * k, x * y * k - z * s, x * z * k + y * s],
        [y * x * k + z * s, c + y * y * k, y * z * k - x * s],
        [z * x * k - y * s, z * y * k + x * s, c + z * z * k],
    ]
}

fn rot_z(angle: f64) -> [[f64; 3]; 3] {
    let (c, s) = (angle.cos(), angle.sin());
    [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
}

/// Fit the ground plane to the lowest band of points and return the rotation that
/// makes its normal point straight up (+z). This levels the scan automatically.
fn level_rotation(pts: &[[f32; 4]]) -> [[f64; 3]; 3] {
    let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let zmin = pts.iter().map(|p| p[2]).fold(f32::INFINITY, f32::min) as f64;
    let zmax = pts.iter().map(|p| p[2]).fold(f32::NEG_INFINITY, f32::max) as f64;
    let cutoff = zmin + 0.3 * (zmax - zmin); // lowest 30% = the floor/ground

    // least squares fit of z = a*x + b*y + c over the ground band
    let (mut sxx, mut sxy, mut sx, mut syy, mut sy, mut n) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    let (mut sxz, mut syz, mut sz) = (0.0, 0.0, 0.0);
    for p in pts {
        let (x, y, z) = (p[0] as f64, p[1] as f64, p[2] as f64);
        if z > cutoff {
            continue;
        }
        sxx += x * x;
        sxy += x * y;
        sx += x;
        syy += y * y;
        sy += y;
        sxz += x * z;
        syz += y * z;
        sz += z;
        n += 1.0;
    }
    if n < 50.0 {
        return identity;
    }
    let m = [[sxx, sxy, sx], [sxy, syy, sy], [sx, sy, n]];
    match solve3(&m, [sxz, syz, sz]) {
        Some([a, b, _]) => {
            let len = (a * a + b * b + 1.0).sqrt();
            rotation_between([-a / len, -b / len, 1.0 / len], [0.0, 0.0, 1.0])
        }
        None => identity,
    }
}

/// Solve a 3x3 linear system by Cramer's rule. None if (near) singular.
fn solve3(m: &[[f64; 3]; 3], r: [f64; 3]) -> Option<[f64; 3]> {
    let det = |a: &[[f64; 3]; 3]| {
        a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
    };
    let d = det(m);
    if d.abs() < 1e-9 {
        return None;
    }
    let mut out = [0.0; 3];
    for col in 0..3 {
        let mut mc = *m;
        for row in 0..3 {
            mc[row][col] = r[row];
        }
        out[col] = det(&mc) / d;
    }
    Some(out)
}

/// Find the yaw (about +z) that best snaps the leveled points to the grid, by
/// minimising the axis-aligned bounding-box area of the horizontal footprint.
fn best_yaw(pts: &[[f32; 4]], level: &[[f64; 3]; 3]) -> f64 {
    let horiz: Vec<(f64, f64)> = pts
        .iter()
        .map(|p| {
            let v = rotate3(level, [p[0] as f64, p[1] as f64, p[2] as f64]);
            (v[0], v[1])
        })
        .collect();
    let mut best_angle = 0.0;
    let mut best_area = f64::INFINITY;
    let mut deg = 0.0_f64;
    while deg < 90.0 {
        let a = deg.to_radians();
        let (c, s) = (a.cos(), a.sin());
        let (mut xmin, mut xmax, mut ymin, mut ymax) =
            (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
        for &(x, y) in &horiz {
            let rx = c * x - s * y;
            let ry = s * x + c * y;
            xmin = xmin.min(rx);
            xmax = xmax.max(rx);
            ymin = ymin.min(ry);
            ymax = ymax.max(ry);
        }
        let area = (xmax - xmin) * (ymax - ymin);
        if area < best_area {
            best_area = area;
            best_angle = a;
        }
        deg += 1.0;
    }
    best_angle
}

fn vec_lerp(a: [f64; 3], b: [f64; 3], alpha: f64) -> [f64; 3] {
    [
        a[0] + alpha * (b[0] - a[0]),
        a[1] + alpha * (b[1] - a[1]),
        a[2] + alpha * (b[2] - a[2]),
    ]
}

fn quat_lerp(a: [f64; 4], mut b: [f64; 4], alpha: f64) -> [f64; 4] {
    let dot: f64 = (0..4).map(|i| a[i] * b[i]).sum();
    if dot < 0.0 {
        b = [-b[0], -b[1], -b[2], -b[3]];
    }
    let mut r = [0.0f64; 4];
    for i in 0..4 {
        r[i] = a[i] + alpha * (b[i] - a[i]);
    }
    let n = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2] + r[3] * r[3]).sqrt();
    let n = if n == 0.0 { 1.0 } else { n };
    [r[0] / n, r[1] / n, r[2] / n, r[3] / n]
}

fn point_fields() -> Vec<PackedElementField> {
    let f = |name: &str, offset: u32, ty: NumericType| PackedElementField {
        name: name.to_string(),
        offset,
        r#type: ty as i32,
    };
    use NumericType::{Float32, Uint8};
    vec![
        f("x", 0, Float32),
        f("y", 4, Float32),
        f("z", 8, Float32),
        f("red", 12, Uint8),
        f("green", 13, Uint8),
        f("blue", 14, Uint8),
        f("alpha", 15, Uint8),
        f("distance", 16, Float32),
    ]
}

fn push_point(buf: &mut Vec<u8>, x: f32, y: f32, z: f32, dist: f32) {
    buf.extend_from_slice(&x.to_le_bytes());
    buf.extend_from_slice(&y.to_le_bytes());
    buf.extend_from_slice(&z.to_le_bytes());
    let t = (((z as f64) - COLOR_LO) / (COLOR_HI - COLOR_LO)).clamp(0.0, 1.0);
    buf.push((255.0 * t) as u8);
    buf.push(40);
    buf.push((255.0 * (1.0 - t)) as u8);
    buf.push(255);
    buf.extend_from_slice(&dist.to_le_bytes());
}

fn identity_pose() -> Pose {
    Pose {
        position: Some(Vector3 { x: 0.0, y: 0.0, z: 0.0 }),
        orientation: Some(Quaternion { x: 0.0, y: 0.0, z: 0.0, w: 1.0 }),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new("KSP LiDAR -> Foxglove", KRPC_HOST, 50000, 50001)?;
    println!("Connected to kRPC.");
    let sc = SpaceCenter::new(client.clone());
    let lidar = LiDAR::new(client.clone());

    let vessel = sc.get_active_vessel()?;
    let parts = vessel.get_parts()?;

    // find the laser part (by tag, else by LiDAR/LaserDist module)
    let mut laser_part = None;
    if let Ok(mut v) = parts.with_tag(TAG.to_string()) {
        laser_part = v.pop();
    }
    if laser_part.is_none() {
        for m in LIDAR_MODULES {
            if let Ok(mut v) = parts.with_module(m.to_string()) {
                if let Some(p) = v.pop() {
                    laser_part = Some(p);
                    break;
                }
            }
        }
    }
    let laser_part = laser_part.ok_or("No LiDAR part found (tag or LaserDist module)")?;
    println!("Laser part: {}", laser_part.get_title()?);

    // enable the lidar module, keep its handle, and grab its bend limit
    let mut bend_limit = BEND_LIMIT;
    let mut lidar_module = None;
    for module in laser_part.get_modules()? {
        let name = module.get_name()?;
        if LIDAR_MODULES.contains(&name.as_str()) {
            module.set_field_bool("Enabled".to_string(), true)?;
            module.set_field_float("Tick Skip".to_string(), TICK_SPACING)?;
            if bend_limit <= 0.0 {
                if let Ok(s) = module.get_field("Max Bend Y".to_string()) {
                    bend_limit = s.parse::<f64>().unwrap_or(45.0).abs();
                }
            }
            lidar_module = Some(module);
            break;
        }
    }
    let lidar_module = lidar_module.ok_or("laser part has no LiDAR/LaserDist module")?;
    if bend_limit <= 0.0 {
        bend_limit = 45.0;
    }
    println!("Vertical bend '{BEND_FIELD}' sweep +/-{bend_limit:.1} deg");

    // best-effort: spin the rotor
    'rotor: for part in parts.get_all()? {
        for module in part.get_modules()? {
            if module.get_name()?.to_lowercase().contains("rotor") {
                for field in ["RPM Limit", "Target RPM", "RPM"] {
                    if module
                        .set_field_float(field.to_string(), ROTOR_RPM as f32)
                        .is_ok()
                    {
                        println!("Rotor: set '{field}' = {ROTOR_RPM}");
                        break;
                    }
                }
                break 'rotor;
            }
        }
    }

    let laser = lidar.laser(&laser_part)?;
    let flight = vessel.flight(None)?;

    let _server = foxglove::WebSocketServer::new()
        .name("KSP LiDAR")
        .bind("0.0.0.0", 8765)
        .start_blocking()?;
    println!("Foxglove: ws://<this-machine-ip>:8765  fixed frame: {FRAME}");
    println!("  point cloud: /ksp/lidar/points");
    println!("  vehicle box: /ksp/vehicle");
    println!("Scanning. Ctrl+C to stop.");

    let half_period = SWEEP_REVS * (60.0 / ROTOR_RPM);
    let z_sign = if FLIP_Z { -1.0 } else { 1.0 };
    let mut rot = cloud_rotation();
    let mut calibrated = !AUTO_ALIGN;
    let period = std::time::Duration::from_secs_f64(1.0 / PUBLISH_HZ);
    let start = std::time::Instant::now();

    let mut data: Vec<u8> = Vec::new();
    let mut seen: HashSet<(i64, i64, i64)> = HashSet::new();
    let mut raw: Vec<[f32; 4]> = Vec::new(); // unrotated relative points, for calibration
    let mut origin: Option<(f64, f64, f64)> = None;
    let mut vehicle: Option<[f64; 3]> = None;
    let mut smooth_pos: Option<[f64; 3]> = None;
    let mut smooth_quat: Option<[f64; 4]> = None;
    let mut count: usize = 0;

    loop {
        let t = start.elapsed().as_secs_f64();
        let bend = triangle(t, half_period, -bend_limit, bend_limit) as f32;
        let _ = lidar_module.set_field_float(BEND_FIELD.to_string(), bend);

        let cloud = laser.get_cloud().unwrap_or_default();
        if cloud.len() >= 6 {
            let (sx, sy, sz) = (cloud[0], cloud[1], cloud[2]);
            let o = *origin.get_or_insert((sx, sy, sz));
            vehicle = Some(rotate3(&rot, [sx - o.0, sy - o.1, z_sign * (sz - o.2)]));

            let mut i = 3;
            while i + 2 < cloud.len() {
                let (hx, hy, hz) = (cloud[i], cloud[i + 1], cloud[i + 2]);
                i += 3;
                if !(hx.is_finite() && hy.is_finite() && hz.is_finite()) {
                    continue;
                }
                // relative point in the raw (unrotated) cloud frame
                let (xr, yr, zr) = (hx - o.0, hy - o.1, z_sign * (hz - o.2));
                let dist =
                    ((hx - sx).powi(2) + (hy - sy).powi(2) + (hz - sz).powi(2)).sqrt();
                if dist < MIN_RANGE || (MAX_RANGE > 0.0 && dist > MAX_RANGE) {
                    continue;
                }
                let key = (
                    (xr / VOXEL).round() as i64,
                    (yr / VOXEL).round() as i64,
                    (zr / VOXEL).round() as i64,
                );
                if count >= MAX_POINTS || !seen.insert(key) {
                    continue;
                }
                if !calibrated {
                    raw.push([xr as f32, yr as f32, zr as f32, dist as f32]);
                }
                let [x, y, z] = rotate3(&rot, [xr, yr, zr]);
                push_point(&mut data, x as f32, y as f32, z as f32, dist as f32);
                count += 1;
            }

            // once enough of the scan exists, fit "up" (and a grid yaw) and bake
            // the alignment in, then rebuild the published cloud once.
            if !calibrated && raw.len() >= CALIB_MIN_POINTS {
                let zmin = raw.iter().map(|p| p[2]).fold(f32::INFINITY, f32::min);
                let zmax = raw.iter().map(|p| p[2]).fold(f32::NEG_INFINITY, f32::max);
                if (zmax - zmin) as f64 >= CALIB_MIN_RANGE {
                    let level = level_rotation(&raw);
                    let yaw = if AUTO_YAW { best_yaw(&raw, &level) } else { 0.0 };
                    rot = mat_mul(&cloud_rotation(), &mat_mul(&rot_z(yaw), &level));
                    calibrated = true;
                    println!("Auto-aligned: leveled + yaw {:.1} deg", yaw.to_degrees());

                    data.clear();
                    count = 0;
                    for p in &raw {
                        let [x, y, z] = rotate3(&rot, [p[0] as f64, p[1] as f64, p[2] as f64]);
                        push_point(&mut data, x as f32, y as f32, z as f32, p[3]);
                        count += 1;
                    }
                    raw = Vec::new(); // no longer needed
                }
            }
        }

        let stamp = Timestamp::now();

        if count > 0 {
            CLOUD.log(&PointCloud {
                timestamp: Some(stamp.clone()),
                frame_id: FRAME.to_string(),
                pose: Some(identity_pose()),
                point_stride: 20,
                fields: point_fields(),
                data: data.clone().into(),
            });
        }

        if let Some(v) = vehicle {
            let mut orient = [0.0, 0.0, 0.0, 1.0];
            if BOX_TRACK_ORIENTATION {
                if let (Ok(h), Ok(p), Ok(r)) =
                    (flight.get_heading(), flight.get_pitch(), flight.get_roll())
                {
                    orient = euler_to_quat(h as f64, p as f64, r as f64);
                }
            }
            let sp = match smooth_pos {
                None => v,
                Some(prev) => vec_lerp(prev, v, BOX_SMOOTHING),
            };
            let sq = match smooth_quat {
                None => orient,
                Some(prev) => quat_lerp(prev, orient, BOX_SMOOTHING),
            };
            smooth_pos = Some(sp);
            smooth_quat = Some(sq);
            VEHICLE.log(&vehicle_box(&stamp, sp, sq));
        }

        std::thread::sleep(period);
    }
}

#[allow(clippy::too_many_arguments)]
fn vehicle_box(stamp: &Timestamp, pos: [f64; 3], q: [f64; 4]) -> SceneUpdate {
    SceneUpdate {
        deletions: vec![],
        entities: vec![SceneEntity {
            timestamp: Some(stamp.clone()),
            frame_id: FRAME.to_string(),
            id: "vehicle".to_string(),
            cubes: vec![CubePrimitive {
                pose: Some(Pose {
                    position: Some(Vector3 { x: pos[0], y: pos[1], z: pos[2] }),
                    orientation: Some(Quaternion { x: q[0], y: q[1], z: q[2], w: q[3] }),
                }),
                size: Some(Vector3 {
                    x: BOX_SIZE[0],
                    y: BOX_SIZE[1],
                    z: BOX_SIZE[2],
                }),
                color: Some(Color { r: 1.0, g: 0.55, b: 0.0, a: 0.6 }),
            }],
            ..Default::default()
        }],
    }
}
