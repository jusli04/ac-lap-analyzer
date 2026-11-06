"""
ac_telemetry.py
Reads Assetto Corsa shared memory on Windows using mmap + ctypes.
AC exposes three shared memory blocks:
  - Local Car Physics  → "Local Car Physics"
  - Graphics Info      → "Local Car Graphics"
  - Static Info        → "Local Car Statics"
Run this while AC is running (in a session, not the main menu).
"""

import mmap
import ctypes
import time
import struct
from dataclasses import dataclass, field
from typing import List


# ── Shared memory structure definitions ──────────────────────────────────────
# Based on AC SDK docs: https://assettocorsamods.net/threads/doc-shared-memory.166/

class ACPhysics(ctypes.Structure):
    """Live physics data — updated every physics tick (~333Hz)"""
    _fields_ = [
        ("packetId",          ctypes.c_int),
        ("gas",               ctypes.c_float),   # throttle 0-1
        ("brake",             ctypes.c_float),   # brake 0-1
        ("fuel",              ctypes.c_float),
        ("gear",              ctypes.c_int),
        ("rpms",              ctypes.c_int),
        ("steerAngle",        ctypes.c_float),   # steering angle
        ("speedKmh",          ctypes.c_float),   # speed in km/h
        ("velocity",          ctypes.c_float * 3),
        ("accG",              ctypes.c_float * 3),  # G-forces XYZ
        ("wheelSlip",         ctypes.c_float * 4),
        ("wheelLoad",         ctypes.c_float * 4),
        ("wheelsPressure",    ctypes.c_float * 4),
        ("wheelAngularSpeed", ctypes.c_float * 4),
        ("tyreWear",          ctypes.c_float * 4),
        ("tyreDirtyLevel",    ctypes.c_float * 4),
        ("tyreCoreTemp",      ctypes.c_float * 4),
        ("camberRAD",         ctypes.c_float * 4),
        ("suspensionTravel",  ctypes.c_float * 4),
        ("drs",               ctypes.c_float),
        ("tc",                ctypes.c_float),
        ("heading",           ctypes.c_float),
        ("pitch",             ctypes.c_float),
        ("roll",              ctypes.c_float),
        ("cgHeight",          ctypes.c_float),
        ("carDamage",         ctypes.c_float * 5),
        ("numberOfTyresOut",  ctypes.c_int),
        ("pitLimiterOn",      ctypes.c_int),
        ("abs",               ctypes.c_float),
    ]


class ACGraphics(ctypes.Structure):
    """Graphics / session info — updated less frequently"""
    _fields_ = [
        ("packetId",            ctypes.c_int),
        ("status",              ctypes.c_int),    # 0=off, 1=replay, 2=live, 3=pause
        ("session",             ctypes.c_int),
        ("currentTime",         ctypes.c_wchar * 15),
        ("lastTime",            ctypes.c_wchar * 15),
        ("bestTime",            ctypes.c_wchar * 15),
        ("split",               ctypes.c_wchar * 15),
        ("completedLaps",       ctypes.c_int),
        ("position",            ctypes.c_int),
        ("iCurrentTime",        ctypes.c_int),    # current lap time in ms
        ("iLastTime",           ctypes.c_int),    # last lap time in ms
        ("iBestTime",           ctypes.c_int),    # best lap time in ms
        ("sessionTimeLeft",     ctypes.c_float),
        ("distanceTraveled",    ctypes.c_float),
        ("isInPit",             ctypes.c_int),
        ("currentSectorIndex",  ctypes.c_int),
        ("lastSectorTime",      ctypes.c_int),
        ("numberOfLaps",        ctypes.c_int),
        ("tyreCompound",        ctypes.c_wchar * 33),
        ("replayTimeMultiplier",ctypes.c_float),
        ("normalizedCarPosition", ctypes.c_float),  # 0-1 track position
        ("carCoordinates",      ctypes.c_float * 3), # world XYZ position
    ]


class ACStatic(ctypes.Structure):
    """Static info — set once at session start"""
    _fields_ = [
        ("smVersion",       ctypes.c_wchar * 15),
        ("acVersion",       ctypes.c_wchar * 15),
        ("numberOfSessions",ctypes.c_int),
        ("numCars",         ctypes.c_int),
        ("carModel",        ctypes.c_wchar * 33),
        ("track",           ctypes.c_wchar * 33),
        ("playerName",      ctypes.c_wchar * 33),
        ("playerSurname",   ctypes.c_wchar * 33),
        ("playerNick",      ctypes.c_wchar * 33),
        ("sectorCount",     ctypes.c_int),
        ("maxTorque",       ctypes.c_float),
        ("maxPower",        ctypes.c_float),
        ("maxRpm",          ctypes.c_int),
        ("maxFuel",         ctypes.c_float),
        ("suspensionMaxTravel", ctypes.c_float * 4),
        ("tyreRadius",      ctypes.c_float * 4),
    ]


# ── Shared memory reader ──────────────────────────────────────────────────────

class ACSharedMemory:
    """Opens AC shared memory blocks and reads current values."""

    PHYSICS_NAME  = "acpmf_physics"
    GRAPHICS_NAME = "acpmf_graphics"
    STATIC_NAME   = "acpmf_static"

    def __init__(self):
        self._phys_map = None
        self._gfx_map  = None
        self._stat_map  = None

    def connect(self):
        """Open shared memory — call this while AC is running."""
        try:
            self._phys_map = mmap.mmap(-1, ctypes.sizeof(ACPhysics),
                                       tagname=self.PHYSICS_NAME,
                                       access=mmap.ACCESS_READ)
            self._gfx_map  = mmap.mmap(-1, ctypes.sizeof(ACGraphics),
                                       tagname=self.GRAPHICS_NAME,
                                       access=mmap.ACCESS_READ)
            self._stat_map = mmap.mmap(-1, ctypes.sizeof(ACStatic),
                                       tagname=self.STATIC_NAME,
                                       access=mmap.ACCESS_READ)
            return True
        except Exception as e:
            print(f"[AC] Could not connect to shared memory: {e}")
            print("[AC] Make sure Assetto Corsa is running and you are in a session.")
            return False

    def get_physics(self) -> ACPhysics:
        self._phys_map.seek(0)
        buf = self._phys_map.read(ctypes.sizeof(ACPhysics))
        return ACPhysics.from_buffer_copy(buf)

    def get_graphics(self) -> ACGraphics:
        self._gfx_map.seek(0)
        buf = self._gfx_map.read(ctypes.sizeof(ACGraphics))
        return ACGraphics.from_buffer_copy(buf)

    def get_static(self) -> ACStatic:
        self._stat_map.seek(0)
        buf = self._stat_map.read(ctypes.sizeof(ACStatic))
        return ACStatic.from_buffer_copy(buf)

    def close(self):
        for m in [self._phys_map, self._gfx_map, self._stat_map]:
            if m:
                m.close()


# ── Telemetry sample ─────────────────────────────────────────────────────────

@dataclass
class TelemetrySample:
    timestamp_ms:    int
    lap_time_ms:     int
    track_position:  float   # 0-1 normalized
    speed_kmh:       float
    throttle:        float   # 0-1
    brake:           float   # 0-1
    steer_angle:     float   # radians
    gear:            int
    rpms:            int
    g_lat:           float   # lateral G
    g_lon:           float   # longitudinal G
    x:               float   # world X
    z:               float   # world Z (use as 2D Y for track map)
    lap_number:      int


# ── Live recorder ─────────────────────────────────────────────────────────────

class LapRecorder:
    """
    Records telemetry samples lap by lap.
    Detects lap boundaries via completedLaps counter.
    """

    def __init__(self, sample_rate_hz: int = 20):
        self.sample_interval = 1.0 / sample_rate_hz
        self.ac = ACSharedMemory()
        self.laps: List[List[TelemetrySample]] = []
        self.lap_times_ms: List[int] = []   # AC-reported lap times, parallel to self.laps
        self._current_lap: List[TelemetrySample] = []
        self._last_lap_count = -1

    def record(self, duration_seconds: float = 300):
        """Record telemetry for up to duration_seconds. Ctrl+C to stop early."""
        if not self.ac.connect():
            return

        static = self.ac.get_static()
        car   = static.carModel.strip() or "unknown"
        track = static.track.strip()    or "unknown"
        print(f"[AC] Connected — Car: {car}  Track: {track}")
        print(f"[AC] Recording at {int(1/self.sample_interval)}Hz. Press Ctrl+C to stop.\n")

        PIT_EXIT_SPEED = 80  # km/h — above this means we've left the pit lane
        _exited_pits = False

        start = time.time()
        try:
            while time.time() - start < duration_seconds:
                t0 = time.time()

                phys = self.ac.get_physics()
                gfx  = self.ac.get_graphics()

                # Wait for pit exit before we start recording anything
                if not _exited_pits:
                    if phys.speedKmh > PIT_EXIT_SPEED:
                        _exited_pits = True
                        self._last_lap_count = gfx.completedLaps
                        print(f"[AC] Pit exit detected at {phys.speedKmh:.0f} km/h — lap recording started.")
                    else:
                        time.sleep(0.05)
                        continue

                # Detect lap change
                if gfx.completedLaps > self._last_lap_count:
                    if self._current_lap:
                        # iLastTime is set by AC exactly at lap boundary — read it now
                        ac_lap_time = gfx.iLastTime
                        self.laps.append(self._current_lap)
                        self.lap_times_ms.append(ac_lap_time)
                        lap_idx = len(self.laps)
                        print(f"[AC] Lap {lap_idx} saved — {ac_lap_time/1000:.3f}s  ({len(self._current_lap)} samples)")
                    self._current_lap = []
                    self._last_lap_count = gfx.completedLaps

                sample = TelemetrySample(
                    timestamp_ms   = int((time.time() - start) * 1000),
                    lap_time_ms    = gfx.iCurrentTime,
                    track_position = gfx.normalizedCarPosition,
                    speed_kmh      = phys.speedKmh,
                    throttle       = phys.gas,
                    brake          = phys.brake,
                    steer_angle    = phys.steerAngle,
                    gear           = phys.gear,
                    rpms           = phys.rpms,
                    g_lat          = phys.accG[0],
                    g_lon          = phys.accG[2],
                    x              = gfx.carCoordinates[0],
                    z              = gfx.carCoordinates[2],
                    lap_number     = gfx.completedLaps + 1,
                )
                self._current_lap.append(sample)

                elapsed = time.time() - t0
                sleep_t = self.sample_interval - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)

        except KeyboardInterrupt:
            print("\n[AC] Recording stopped.")
            print(f"[AC] Partial lap discarded ({len(self._current_lap)} samples — complete a lap before stopping to save it)")

        finally:
            self.ac.close()

        print(f"[AC] Total laps recorded: {len(self.laps)}")
        return self.laps