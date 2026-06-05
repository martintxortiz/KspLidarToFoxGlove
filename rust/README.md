# KSP LiDAR -> Foxglove Bridge (Rust)

A native Rust port of the Python bridge. Same behaviour (accumulating,
height-coloured 3D scan + a vehicle box that tracks pose), but the hot per-point
loop runs compiled instead of interpreted.

## Build & run

Needs the Rust toolchain (`cargo`). Run from **this `rust/` directory** so the
build picks up `.cargo/config.toml`, which points the kRPC code generator at the
service definitions in `services/`:

```powershell
cd C:\Users\Lowgo\Code\ksp\foxglove_lidar_bridge\rust
cargo run --release
```

The first build takes a couple of minutes (it compiles the Foxglove SDK and
generates the kRPC client from the JSON service definitions). After that, the
binary is at `target\release\ksp_lidar_bridge.exe`.

> Stop the Python bridge first — both listen on `ws://127.0.0.1:8765`, and only
> one can hold the port (otherwise you get `AddrInUse`).

Foxglove:

```text
Connection:   ws://127.0.0.1:8765
Fixed frame:  world
Point cloud:  /ksp/lidar/points   (3D panel)
Vehicle box:  /ksp/vehicle        (same 3D panel)
```

## Settings

All knobs are `const`s at the top of `src/main.rs` — same names as the Python
version (`TAG`, `ROTOR_RPM`, `SWEEP_REVS`, `BEND_FIELD`, `BEND_LIMIT`, `VOXEL`,
`MIN_RANGE`/`MAX_RANGE`, `FLIP_Z`, `BOX_SIZE`, `BOX_TRACK_ORIENTATION`,
`YAW_OFFSET_DEG`, `BOX_SMOOTHING`). Edit and re-run `cargo run --release`.

### Aligning the cloud with the world grid (automatic)

The cloud arrives in KSP's world axes, which sit twisted/tilted relative to the
local horizon, so the scanned planes can look rotated vs the (world-aligned) box.

With `AUTO_ALIGN = true` (default) this fixes itself: once `CALIB_MIN_POINTS` of
scan have accumulated, the bridge fits the **ground plane** (the lowest ~30 % of
points) to find true "up" and levels the cloud, then — with `AUTO_YAW = true` —
finds the yaw that **snaps the walls to the grid** (minimum bounding-box area)
and bakes both in, rebuilding the cloud once. You'll see `Auto-aligned: leveled
+ yaw N deg` print when it happens. The box is rotated by the same transform, so
it stays stuck to the rover.

No tuning needed. The `CLOUD_YAW_DEG` / `CLOUD_PITCH_DEG` / `CLOUD_ROLL_DEG`
consts remain as an optional extra nudge applied on top (default 0). To go back
to fully manual, set `AUTO_ALIGN = false`.

## How the kRPC bindings are generated

`services/` holds `KRPC.Core.json`, `KRPC.SpaceCenter.json`, and
`KRPC.LiDAR.json` (copied from `GameData/kRPC`). `krpc-client`'s build script
reads `KRPC_SERVICES` (set to `services/` by `.cargo/config.toml`) and generates
a typed Rust client, including the custom `LiDAR` service. If you update KSP/kRPC
and the API changes, recopy those JSONs and rebuild.
