# -*- coding: utf-8 -*-
"""
Strapping Table ("Cetvel") Interpolation Model - Educational Prototype
======================================================================
A horizontal cylindrical fuel tank has a NON-LINEAR level-to-volume
relationship: the same 1 cm level change corresponds to many liters
near the middle of the tank, but far fewer liters near the bottom/top.

The ATG/automation system therefore stores a STRAPPING TABLE (cetvel):
discrete calibration points (level_cm -> volume_L). Intermediate levels
are estimated by LINEAR INTERPOLATION between the two nearest points.

This script:
  1) computes the EXACT level->volume curve from tank geometry
     (circular-segment formula),
  2) builds a strapping table by sampling that curve at fixed steps,
  3) implements the linear interpolation used by the system,
  4) measures the interpolation (estimation) error, and
  5) demonstrates why equal level changes != equal volume changes.

Author: Ali Edip Mangtay - Internship Day 4 prototype
"""

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1) Tank geometry (horizontal cylinder) and the EXACT volume model
# ----------------------------------------------------------------------
R = 1.20   # tank radius  (m)  -> diameter 2.4 m
L = 8.00   # tank length  (m)  -> full volume ~ 36,000 L

def exact_volume_liters(h_m):
    """Exact fuel volume (liters) at fuel level h (meters).

    Cross-section of the fuel is a CIRCULAR SEGMENT; its area is:
        A(h) = R^2 * arccos((R-h)/R) - (R-h) * sqrt(2*R*h - h^2)
    Volume = A(h) * tank_length. (1 m^3 = 1000 L)
    """
    h = np.clip(np.asarray(h_m, dtype=float), 0.0, 2.0 * R)
    seg_area = (R**2 * np.arccos((R - h) / R)
                - (R - h) * np.sqrt(np.maximum(2.0 * R * h - h**2, 0.0)))
    return seg_area * L * 1000.0


# ----------------------------------------------------------------------
# 2) Build the strapping table (the "cetvel"): discrete calibration rows
# ----------------------------------------------------------------------
def build_strapping_table(step_cm):
    """Sample the exact curve every `step_cm` centimeters -> (levels_m, volumes_L)."""
    levels = np.arange(0.0, 2.0 * R + 1e-9, step_cm / 100.0)
    return levels, exact_volume_liters(levels)


# ----------------------------------------------------------------------
# 3) Linear interpolation between calibration points (what the ATG does)
# ----------------------------------------------------------------------
def interp_volume_liters(h_m, tbl_levels, tbl_volumes):
    """Estimate volume at level h by LINEAR interpolation on the table.

    Own implementation (verified against np.interp below):
    find the bracketing rows (h_lo, v_lo) and (h_hi, v_hi), then
        v = v_lo + (h - h_lo) * (v_hi - v_lo) / (h_hi - h_lo)
    """
    h = np.clip(np.asarray(h_m, dtype=float), tbl_levels[0], tbl_levels[-1])
    idx = np.clip(np.searchsorted(tbl_levels, h, side="right") - 1,
                  0, len(tbl_levels) - 2)
    h_lo, h_hi = tbl_levels[idx], tbl_levels[idx + 1]
    v_lo, v_hi = tbl_volumes[idx], tbl_volumes[idx + 1]
    return v_lo + (h - h_lo) * (v_hi - v_lo) / (h_hi - h_lo)


if __name__ == "__main__":
    # --- sanity check: our interpolation must match numpy's reference ---
    tbl_h, tbl_v = build_strapping_table(step_cm=10)          # 10 cm table
    probe = np.linspace(0, 2 * R, 4001)
    assert np.allclose(interp_volume_liters(probe, tbl_h, tbl_v),
                       np.interp(probe, tbl_h, tbl_v)), "interp mismatch!"

    # --- estimation error of the table vs the exact physical curve ------
    exact = exact_volume_liters(probe)
    err_10cm = interp_volume_liters(probe, tbl_h, tbl_v) - exact
    tbl_h25, tbl_v25 = build_strapping_table(step_cm=25)       # coarser table
    err_25cm = interp_volume_liters(probe, tbl_h25, tbl_v25) - exact

    print(f"Tank: R={R} m, L={L} m, full volume = {exact_volume_liters(2*R):,.0f} L")
    print(f"10 cm table -> {len(tbl_h):3d} rows | max |error| = {np.max(np.abs(err_10cm)):6.1f} L")
    print(f"25 cm table -> {len(tbl_h25):3d} rows | max |error| = {np.max(np.abs(err_25cm)):6.1f} L")

    # --- why equal level changes are NOT equal volume changes ----------
    dh = 0.01  # a 1 cm level change...
    for h0 in (0.10, 1.20, 2.30):  # near bottom, middle, near top
        dv = exact_volume_liters(h0 + dh) - exact_volume_liters(h0)
        print(f"  1 cm change at level {h0:4.2f} m  ->  {dv:6.1f} L")

    # --------------------------- plots ---------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    ax.plot(probe, exact, lw=2, label="Exact geometry curve")
    ax.plot(tbl_h, tbl_v, "o", ms=4, label="Strapping table rows (10 cm)")
    ax.set_xlabel("Fuel level (m)"); ax.set_ylabel("Volume (L)")
    ax.set_title("Level → Volume is non-linear"); ax.legend()

    ax = axes[1]
    levels_demo = np.array([0.10, 1.20, 2.30])
    dv_demo = exact_volume_liters(levels_demo + dh) - exact_volume_liters(levels_demo)
    ax.bar(["bottom\n0.10 m", "middle\n1.20 m", "top\n2.30 m"], dv_demo)
    ax.set_ylabel("Liters per 1 cm level change")
    ax.set_title("Same Δlevel ≠ same Δvolume")

    ax = axes[2]
    ax.plot(probe, err_10cm, lw=1.5, label="10 cm table error")
    ax.plot(probe, err_25cm, lw=1.5, label="25 cm table error")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Fuel level (m)"); ax.set_ylabel("Interpolation error (L)")
    ax.set_title("Estimation error vs table resolution"); ax.legend()

    plt.tight_layout()
    plt.savefig("strapping_table_model.png", dpi=130)
    print("\nSaved figure -> strapping_table_model.png")
    plt.show()
