#!/usr/bin/env python3
"""Stream KSP LiDAR points to Foxglove as an accumulating, height-colored 3D scan.

The laser sits on a motor spinning at ROTOR_RPM (horizontal coverage). This
script sweeps the laser's vertical bend (Bend Y) across its full range, timed to
the rotor speed, so each revolution scans a slightly different vertical angle --
the result is a full helical 3D point cloud instead of a single flat ring. A box
marker is published at the vehicle's current position.
"""

import math
import struct
import time

import foxglove
import krpc
from foxglove.channels import PointCloudChannel, SceneUpdateChannel
from foxglove.messages import (
    Color,
    CubePrimitive,
    PackedElementField,
    PackedElementFieldNumericType,
    PointCloud,
    Pose,
    Quaternion,
    SceneEntity,
    SceneUpdate,
    Timestamp,
    Vector3,
)

# --- settings ---------------------------------------------------------------
# Where KSP/kRPC is running. "127.0.0.1" = same machine as this script.
# Set to the game PC's LAN IP (e.g. "192.168.1.166") to run the bridge remotely.
KRPC_HOST = "192.168.1.166"
TAG = "laser"  # part tag you set in the editor
TICK_SPACING = 0.0  # LiDAR "Tick Skip" field
TOPIC = "/ksp/lidar/points"
BOX_TOPIC = "/ksp/vehicle"  # vehicle box marker
FRAME = "world"  # Foxglove fixed + display frame
PUBLISH_HZ = 10
FLIP_Z = True  # KSP's up is inverted vs Foxglove; negate Z to fix

VOXEL = 0.1  # metres; points closer than this are merged (dedupe)
MAX_POINTS = 500_000  # stop accumulating past this many points

# range gate (metres): drop hits nearer than MIN or farther than MAX. 0 = no limit.
# MIN clears the laser hitting its own mount; MAX clears far "missed" returns.
MIN_RANGE = 0.0
MAX_RANGE = 0.0

# vehicle box marker (metres); placed at the laser/sensor position
BOX_SIZE = (3.0, 3.0, 1.5)
BOX_TRACK_ORIENTATION = True  # rotate the box with the vehicle (kRPC heading/pitch/roll)
YAW_OFFSET_DEG = 0.0  # rotate box heading to line it up with the scan (tune once)
# low-pass smoothing for the box pose; lower = smoother/steadier but laggier (0..1).
# Also averages out the spinning-mount wobble toward the rover centre.
BOX_SMOOTHING = 0.15

# vertical bend sweep -- this is what gives you the 3rd (vertical) dimension
ROTOR_RPM = 120.0  # speed the top motor spins at
SWEEP_REVS = 0.5  # rotor revolutions per one min->max bend pass
BEND_FIELD = "Bend Y"  # vertical bend field on the laser module
BEND_LIMIT = 0.0  # sweep degrees +/-; 0 = read "Max Bend Y" from the part

# best-effort: spin the rotor from here too (otherwise set its RPM in-game)
CONTROL_ROTOR = True
ROTOR_RPM_FIELDS = ("RPM Limit", "Target RPM", "RPM")  # tried in order

# height -> colour range (blue at COLOR_LO, red at COLOR_HI), metres from start
COLOR_LO = -2.0
COLOR_HI = 20.0

LIDAR_MODULES = ("LiDARModule", "LaserDistModule")
POINT_STRIDE = 20
POINT_FIELDS = [
    PackedElementField(name="x", offset=0, type=PackedElementFieldNumericType.Float32),
    PackedElementField(name="y", offset=4, type=PackedElementFieldNumericType.Float32),
    PackedElementField(name="z", offset=8, type=PackedElementFieldNumericType.Float32),
    PackedElementField(name="red", offset=12, type=PackedElementFieldNumericType.Uint8),
    PackedElementField(
        name="green", offset=13, type=PackedElementFieldNumericType.Uint8
    ),
    PackedElementField(
        name="blue", offset=14, type=PackedElementFieldNumericType.Uint8
    ),
    PackedElementField(
        name="alpha", offset=15, type=PackedElementFieldNumericType.Uint8
    ),
    # true laser range: distance from the sensor when this point was measured
    PackedElementField(
        name="distance", offset=16, type=PackedElementFieldNumericType.Float32
    ),
]
IDENTITY_POSE = Pose(
    position=Vector3(x=0.0, y=0.0, z=0.0),
    orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
)


def triples(values):
    """Group a flat [x, y, z, x, y, z, ...] list into (x, y, z) tuples."""
    return list(zip(values[0::3], values[1::3], values[2::3]))


def triangle(t, half_period, lo, hi):
    """Triangle wave: lo -> hi over half_period seconds, then back, repeating."""
    phase = (t % (2.0 * half_period)) / half_period  # 0..2
    frac = phase if phase <= 1.0 else 2.0 - phase  # 0..1..0
    return lo + (hi - lo) * frac


def pack_point(x, y, z, dist):
    """20 bytes: xyz float32 + rgba (height-coloured) + true laser range float32."""
    t = (z - COLOR_LO) / (COLOR_HI - COLOR_LO)
    t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
    return struct.pack(
        "<fffBBBBf", x, y, z, int(255 * t), 40, int(255 * (1 - t)), 255, dist
    )


def euler_to_quat(heading_deg, pitch_deg, roll_deg):
    """kRPC heading/pitch/roll -> quaternion (x, y, z, w) in a z-up cloud frame."""
    psi = math.radians(YAW_OFFSET_DEG - heading_deg)  # yaw about up (+z)
    theta = math.radians(pitch_deg)
    phi = math.radians(roll_deg)
    cy, sy = math.cos(psi / 2), math.sin(psi / 2)
    cp, sp = math.cos(theta / 2), math.sin(theta / 2)
    cr, sr = math.cos(phi / 2), math.sin(phi / 2)
    return (
        sr * cp * cy - cr * sp * sy,  # x
        cr * sp * cy + sr * cp * sy,  # y
        cr * cp * sy - sr * sp * cy,  # z
        cr * cp * cy + sr * sp * sy,  # w
    )


def vec_lerp(a, b, alpha):
    """Exponential low-pass step between two 3-vectors."""
    return tuple(a[i] + alpha * (b[i] - a[i]) for i in range(3))


def quat_lerp(a, b, alpha):
    """Low-pass step between two quaternions (shortest path, re-normalised)."""
    if sum(a[i] * b[i] for i in range(4)) < 0.0:  # take the shorter arc
        b = tuple(-c for c in b)
    r = tuple(a[i] + alpha * (b[i] - a[i]) for i in range(4))
    n = math.sqrt(sum(c * c for c in r)) or 1.0
    return tuple(c / n for c in r)


def vehicle_box(timestamp, position, orientation):
    """A SceneUpdate with one box at the vehicle pose. orientation is (x, y, z, w)."""
    x, y, z = position
    qx, qy, qz, qw = orientation
    cube = CubePrimitive(
        pose=Pose(
            position=Vector3(x=x, y=y, z=z),
            orientation=Quaternion(x=qx, y=qy, z=qz, w=qw),
        ),
        size=Vector3(x=BOX_SIZE[0], y=BOX_SIZE[1], z=BOX_SIZE[2]),
        color=Color(r=1.0, g=0.55, b=0.0, a=0.6),
    )
    entity = SceneEntity(timestamp=timestamp, frame_id=FRAME, id="vehicle", cubes=[cube])
    return SceneUpdate(entities=[entity])


def find_lasers(conn):
    """Enable matching LiDAR parts; return [(laser_handle, lidar_module), ...]."""
    vessel = conn.space_center.active_vessel

    candidates = []
    try:
        candidates += vessel.parts.with_tag(TAG)
    except Exception:
        pass
    for name in LIDAR_MODULES:
        try:
            candidates += vessel.parts.with_module(name)
        except Exception:
            pass

    lasers = []
    seen = set()
    for part in candidates:
        key = getattr(part, "_object_id", id(part))
        if key in seen:
            continue
        seen.add(key)
        for module in part.modules:
            if module.name in LIDAR_MODULES:
                module.set_field_bool("Enabled", True)
                module.set_field_float("Tick Skip", TICK_SPACING)
                lasers.append((conn.lidar.laser(part), module))
                print(f"  laser on {part.title} (tag={part.tag!r})")
                print(f"    fields: {sorted(module.fields)}")

    if not lasers:
        tags = sorted({p.tag for p in vessel.parts.all if p.tag})
        raise RuntimeError(
            f"No LiDAR parts found (looked for tag={TAG!r} or modules {LIDAR_MODULES}).\n"
            f"Tags present on '{vessel.name}': {tags or '(none)'}\n"
            f"Set TAG at the top of the script to match, or tag the part 'laser' in the editor."
        )
    return lasers


def start_rotor(conn):
    """Best-effort: set the robotic rotor spinning at ROTOR_RPM.

    Note: never reads module.fields -- some rotor modules have duplicate field
    names that make kRPC throw. We set fields by name and swallow failures.
    """
    if not CONTROL_ROTOR:
        return
    for module in (
        m for p in conn.space_center.active_vessel.parts.all for m in p.modules
    ):
        if "rotor" not in module.name.lower():
            continue
        for name in ROTOR_RPM_FIELDS:
            try:
                module.set_field_float(name, ROTOR_RPM)
                print(f"  rotor: set {name!r} = {ROTOR_RPM}")
                return
            except Exception:
                pass
        print(
            f"  found rotor '{module.name}' but couldn't set RPM; set it to {ROTOR_RPM} in-game"
        )
        return
    print(f"  no rotor module found; set its RPM to {ROTOR_RPM} in-game")


def main():
    conn = krpc.connect(
        name="KSP LiDAR -> Foxglove",
        address=KRPC_HOST,
        rpc_port=50000,
        stream_port=50001,
    )
    print("Connected to kRPC.")
    vessel = conn.space_center.active_vessel
    flight = vessel.flight()  # heading/pitch/roll relative to the local surface
    lasers = find_lasers(conn)
    start_rotor(conn)

    bend_field = BEND_FIELD
    bend_limit = BEND_LIMIT
    if bend_limit <= 0.0:  # read the laser's own max bend
        try:
            bend_limit = abs(float(lasers[0][1].get_field("Max Bend Y")))
        except Exception:
            bend_limit = 45.0
    print(f"Vertical bend field: {bend_field!r}  (sweep +/-{bend_limit:.1f} deg)")

    server = foxglove.start_server(
        name="KSP LiDAR", host="0.0.0.0", port=8765, message_backlog_size=1
    )
    channel = PointCloudChannel(TOPIC)
    box_channel = SceneUpdateChannel(BOX_TOPIC)
    print(f"Foxglove: ws://<this-machine-ip>:8765   fixed frame: {FRAME}")
    print(f"  point cloud: {TOPIC}")
    print(f"  vehicle box: {BOX_TOPIC}")
    print("Scanning. Ctrl+C to stop.")

    half_period = SWEEP_REVS * (60.0 / ROTOR_RPM)  # seconds for one min->max pass
    z_sign = -1.0 if FLIP_Z else 1.0
    data = bytearray()  # incrementally packed cloud bytes
    seen_voxels = set()  # occupied voxel keys, for dedupe
    origin = None  # fixed reference captured from the first reading
    vehicle = None  # latest vehicle (sensor) position in the cloud frame
    smooth_pos = None  # low-pass filtered box position
    smooth_quat = None  # low-pass filtered box orientation
    count = 0
    t0 = time.monotonic()

    try:
        while True:
            bend = triangle(time.monotonic() - t0, half_period, -bend_limit, bend_limit)

            for laser, module in lasers:
                if bend_field:
                    try:
                        module.set_field_float(bend_field, bend)
                    except Exception:
                        pass

                raw = triples(list(laser.cloud))
                if len(raw) < 2:
                    continue
                if origin is None:
                    origin = raw[0]
                sx, sy, sz = raw[0]  # sensor position for this reading
                vehicle = (sx - origin[0], sy - origin[1], z_sign * (sz - origin[2]))

                for hx, hy, hz in raw[1:]:
                    x = hx - origin[0]
                    y = hy - origin[1]
                    z = z_sign * (hz - origin[2])
                    if not (x == x and y == y and z == z):  # NaN check
                        continue
                    dist = math.sqrt((hx - sx) ** 2 + (hy - sy) ** 2 + (hz - sz) ** 2)
                    if dist < MIN_RANGE or (MAX_RANGE and dist > MAX_RANGE):
                        continue
                    key = (round(x / VOXEL), round(y / VOXEL), round(z / VOXEL))
                    if key in seen_voxels or count >= MAX_POINTS:
                        continue
                    seen_voxels.add(key)
                    data += pack_point(x, y, z, dist)
                    count += 1

            now = time.time_ns()
            stamp = Timestamp(now // 1_000_000_000, now % 1_000_000_000)

            if count:
                channel.log(
                    PointCloud(
                        timestamp=stamp,
                        frame_id=FRAME,
                        pose=IDENTITY_POSE,
                        point_stride=POINT_STRIDE,
                        fields=POINT_FIELDS,
                        data=bytes(data),
                    )
                )

            if vehicle is not None:
                orientation = (0.0, 0.0, 0.0, 1.0)
                if BOX_TRACK_ORIENTATION:
                    try:
                        orientation = euler_to_quat(
                            flight.heading, flight.pitch, flight.roll
                        )
                    except Exception:
                        pass
                if smooth_pos is None:  # seed the filters on the first sample
                    smooth_pos, smooth_quat = vehicle, orientation
                else:
                    smooth_pos = vec_lerp(smooth_pos, vehicle, BOX_SMOOTHING)
                    smooth_quat = quat_lerp(smooth_quat, orientation, BOX_SMOOTHING)
                box_channel.log(vehicle_box(stamp, smooth_pos, smooth_quat))

            time.sleep(1.0 / PUBLISH_HZ)
    except KeyboardInterrupt:
        print(f"\nStopped. {count} points accumulated.")
    finally:
        conn.close()
        server.stop()


if __name__ == "__main__":
    main()
