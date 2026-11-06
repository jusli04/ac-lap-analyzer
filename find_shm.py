"""
find_shm.py
Scans known Assetto Corsa shared memory block names and reports which ones
are present and have non-zero data. Run this while AC is running on track.
"""
import mmap
import ctypes
import struct
 
# All known AC / CM shared memory block names
CANDIDATES = [
    "Local Car Physics",
    "Local Car Graphics",
    "Local Car Statics",
    "acpmf_physics",
    "acpmf_graphics",
    "acpmf_static",
    "AC_SHARED_MEMORY_PHYSICS",
    "AC_SHARED_MEMORY_GRAPHICS",
    "AC_SHARED_MEMORY_STATIC",
]
 
print("Scanning shared memory blocks (run this while AC is on track)...\n")
 
found_any = False
for name in CANDIDATES:
    for size in [4096, 8192, 16384, 32768, 65536]:
        try:
            m = mmap.mmap(-1, size, tagname=name, access=mmap.ACCESS_READ)
            data = m.read(size)
            m.close()
            nonzero = sum(1 for b in data if b != 0)
            status = "✅ HAS DATA" if nonzero > 10 else "⚠️  exists but empty"
            print(f"  [{status}]  '{name}'  (size={size}, nonzero_bytes={nonzero})")
            found_any = True
            break
        except Exception:
            continue
 
if not found_any:
    print("No shared memory blocks found at all.")
    print("Make sure AC is running and you are IN a session on track.")
 
print("\nDone.")
 