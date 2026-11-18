"""Show that unreal.Rotator's positional order in Python is (roll, pitch, yaw), not the C++ (pitch, yaw, roll).
Run inside Unreal Editor's Python (see README.md). Prints one OK or FAIL line."""
import unreal

r = unreal.Rotator(0, -90, 0)
print(f"Rotator(0, -90, 0): roll={r.roll} pitch={r.pitch} yaw={r.yaw}")
k = unreal.Rotator(roll=0, pitch=0, yaw=-90)
if r.pitch == -90 and r.yaw == 0 and k.yaw == -90 and k.pitch == 0:
    print("OK: positional order is (roll, pitch, yaw); pass keywords")
else:
    print(f"FAIL: positional order is not (roll, pitch, yaw) here; keywords gave pitch={k.pitch} yaw={k.yaw}")
