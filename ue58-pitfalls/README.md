# UE 5.8 pitfalls that fail silently

## Problem

In Unreal Engine 5.8 several editor-Python calls, importers and material settings fail without an error: the call
returns, the log stays clean, and the result is wrong or missing. Each item below cost us hours because nothing
reported it.

## Who this concerns

- Unreal Engine 5.8 on Windows, mostly editor Python (`unreal` module) in the editor or in a headless
  `-run=pythonscript` commandlet, FBX, glTF and VRM import, and material authoring from Python.
- Engine build: we used 5.8.4. The list does not record a build per item, so we cannot say which builds share each one.
- Upstream fixes: none known to us.

## What we changed

We changed nothing in the engine. These are the pitfalls with their symptoms and the workaround we use. The general rule:
assert the effect, not the call. After every editor-Python step, read back what you made and print an `OK` or `FAIL`
line that your runner can find in the log.

### Levels and saving

- **`LevelEditorSubsystem.new_level(path)` on a path that already exists.** Symptom: it fails silently, and
  `save_current_level()` then saves nothing; the game ran a stale empty map for hours. Fix: delete the map asset first,
  then assert that both calls returned true and that the asset exists afterwards.
- **`EditorActorSubsystem.spawn_actor_from_object(mesh, ...)` in a headless editor commandlet.** Symptom: it spawns
  nothing and returns `None`; the log says only `SpawnActorFromObject. No actor was spawned.` Fix: spawn a
  `StaticMeshActor` with `spawn_actor_from_class`, then call `static_mesh_component.set_static_mesh`.

### C++

- **`UE_LOG` before an `else`.** Symptom: `if (x) UE_LOG(...); else ...` fails to compile with C2181 "illegal else
  without matching if", because `UE_LOG` expands to a braced block in 5.8. Fix: wrap a `UE_LOG` in braces whenever an
  `else` follows it.

### Importing

- **FBX import settings ignored.** Symptom: FBX goes through Interchange in 5.8, which ignores `FbxImportUI` (unit
  scale, vertex colours, collision); our world arrived 100x too big with no body setup. Fix: switch FBX to the legacy
  importer with the console variable `Interchange.FeatureFlags.Import.FBX 0` before importing, and create body setups
  explicitly. Check the bounds afterwards.
- **VRM through Interchange glTF.** Symptom: a VRM imports as one skeletal mesh per node. Fix: join the meshes in
  Blender before import.
- **VRM retargeting.** Cause: `ABP_GenericRetarget`'s profile code expects Epic's op stack. Fix: make each VRM retargeter a duplicate of `RTG_UEFN_to_UE5_Mannequin` with the
  target rig swapped and a fresh auto-aligned retarget pose.
- **`unreal.Rotator(a, b, c)` argument order.** Symptom: in Python the positional order is (roll, pitch, yaw), not the
  C++ (pitch, yaw, roll). `Rotator(0, -90, 0)` meant as a yaw pitched an imported mesh onto its side, and a 25k-instance
  crowd rendered as pebbles. Fix: pass keywords, `unreal.Rotator(roll=0, pitch=0, yaw=-90)`, and check the bounds' up
  axis after any rotation. `check_rotator.py` shows the order.
- **Texture size read back in a commandlet.** Symptom: in a `-run=pythonscript` commandlet,
  `Texture2D.blueprint_get_size_x/y` read 32x32 for freshly baked AnimToTexture textures, correct ones included. Fix:
  do not assert the expected dimensions there; log them, and check the values the decode depends on (the data asset's
  rows per frame) instead.

### Materials

- **`MaterialExpressionVertexColor` pin name.** Symptom: it has no `RGB` output; its colour pin is the empty string
  `""`. A wrong pin name connects nothing and reports nothing. Fix: use `""`, and read the connections back after wiring.
- **Usage flags.** Symptom: without `used_with_skeletal_mesh` and `used_with_morph_targets`, the game run from the
  editor recompiles materials at runtime (under 1 fps), and a cooked build shows them grey. Fix: set both flags when
  authoring the material.
- **Material Parameter Collection from Python.** Symptom: a collection made from Python asserted at runtime. Fix: use
  per-material vector parameters instead.
- **VRoid hair.** Cause: VRoid hair is a shell that needs back-face culling. Fix: never make the hair material
  two-sided.

### Rendering outside Unreal (Blender)

- **Workbench renders.** Symptom: Blender's Workbench engine ignores alpha and base colour factors. Fix: rewire base
  colour to emission and render with EEVEE.
- **Blender exit code.** Symptom: Blender exits 0 when a `--python` script raises. Fix: pass `--python-exit-code 1` so a
  failed script fails the run.

### Running and capturing

- **A black screen.** Symptom: the game shows only black. Fix: run `ToggleDebugCamera` from a
  scripted run and print the camera location; in our project it showed the pawn falling through the world. A line trace
  that reports what is under a point and its material proves the collision.
- **Recording Direct3D 12.** Symptom: ffmpeg's `gdigrab` records Direct3D 12 windows as black. Fix: use ffmpeg's
  `gfxcapture` (Windows.Graphics.Capture).
- **Commandlet hangs at shutdown.** Symptom: a commandlet can hang at shutdown after its work is done. Fix: run it under
  a timeout, kill it there, and judge the run by its log.
- **First launch stalls.** Symptom: the first launch sits 2-3 minutes on a black window (asset registry and PSO
  warm-up). Fix: none needed; wait. Later launches are quicker.
- **Packaging drops content.** Symptom: packaging our project needed the Game Animation Sample's
  `DefaultGameplayTags.ini` and a `GameFeatureData` asset-manager rule, and a cook that drops content usually logs it
  only as a warning. Fix: add both, and read the cook's warnings.
- **A build or editor start that seems hung.** Symptom: `cl.exe` or the editor sits at no CPU. On our machine this was
  usually a busy spinning disk (538% disk time, queue length 4). Fix: check with
  `Get-Counter '\PhysicalDisk(*)\% Disk Time'` before killing anything. UBA and `-NoUBA` neither cause nor cure it.

## Result

A list of cases we observed, not a measurement. Each fix above was what made the step work for us.

## How to run

`check_rotator.py` shows the `unreal.Rotator` argument order. It needs Unreal Engine 5.8 with the Python Editor Script
Plugin enabled, and any project.

In the editor: Output Log, switch the input to Python, and run `py "<path to>/check_rotator.py"`.

Headless:

```
"<UE 5.8 root>/Engine/Binaries/Win64/UnrealEditor-Cmd.exe" "<project>.uproject" -run=pythonscript -script="<path to>/check_rotator.py" -nullrhi -unattended -nosplash
```

Expected lines in the log:

```
Rotator(0, -90, 0): roll=0.0 pitch=-90.0 yaw=0.0
OK: positional order is (roll, pitch, yaw); pass keywords
```

The other items need assets, so they have no script here; each item's fix names the read-back that checks it.

## Measured on

- Unreal Engine 5.8.4 (the installed build), Windows 11. The MSVC version for the C++ item was not recorded.
- Blender items: the Blender version was not recorded.

## Not verified

- `check_rotator.py` has not been run in this form.
- Which 5.8 builds share each pitfall, and whether 5.9 fixes any.
- The disk figures come from one observation on one machine.
