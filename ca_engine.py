"""
ca_engine.py
-------------
Core cellular-automata (CA) wildfire spread model.

Cell states: 0 = unburned, 1 = burning, 2 = burnt
Grid resolution matches the static layers (30m Sentinel/Landsat-aligned).

Transition rule (unburned -> burning), per cell i, per timestep:

    P(i) = base_spread_rate
           * neighbor_term(i)      # kitne aur kaunse-direction neighbors jal rahe hain
           * wind_factor(i)        # wind speed ka multiplier
           * dryness_factor(i)     # curing index (0-1, zyada = zyada sookha = zyada risk)
           * lai_factor(i)         # fuel load - zyada LAI thoda zyada fuel (par bahut zyada
                                     LAI kabhi kabhi zyada moisture bhi matlab ho sakta hai -
                                     yahan simplistic linear positive assume kiya hai, calibrate
                                     karna hindcast se)
           * slope_factor(i)       # upward slope spread ko tez karta hai

Wind-direction weighting: har jalte hue neighbor se current cell ki taraf
ka bearing nikalo, wind-direction ke saath uska angular alignment dekho -
downwind spread ko zyada weight milta hai (fire ellipse jaisा simplistic version).

Yeh module dono raster arrays (numpy) pe kaam karta hai - static layers
(static_layers.py se GeoTIFF -> numpy) aur weather (weather_fetch.py se
per-block -> per-cell broadcast) ko align karke caller CAModel.step() call
karta hai.

NOTE: parameters (spread_rate, wind_weight, dryness_weight, etc.) abhi
reasonable defaults hain - hindcast mode inhi ko calibrate/fit karne ke
liye hai (grid search ya simple optimization historical FIRMS progression
ke against).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict


UNBURNED, BURNING, BURNT = 0, 1, 2

# 8-connected neighborhood offsets aur unki compass bearing (degrees, 0=N, 90=E)
NEIGHBOR_OFFSETS = [
    (-1, 0, 0), (-1, 1, 45), (0, 1, 90), (1, 1, 135),
    (1, 0, 180), (1, -1, 225), (0, -1, 270), (-1, -1, 315),
]


@dataclass
class CAParams:
    base_spread_rate: float = 0.15   # base probability per timestep, per burning neighbor
    wind_weight: float = 0.6         # wind speed ka contribution
    wind_alignment_power: float = 2.0  # downwind favor kitna sharp ho (cosine^power)
    dryness_weight: float = 0.8      # curing index ka contribution
    lai_weight: float = 0.2          # fuel load ka contribution
    slope_weight: float = 0.3        # upslope spread boost
    burn_duration_steps: int = 2     # kitne steps tak cell "burning" rehta hai phir "burnt" ho jata hai
    timestep_minutes: int = 60


@dataclass
class GridState:
    state: np.ndarray               # 0/1/2 grid
    burning_since: np.ndarray       # kitne steps se burning hai (burnt transition ke liye)


class CAModel:
    def __init__(self, curing: np.ndarray, lai: np.ndarray, slope: np.ndarray,
                 aspect: np.ndarray, fuel_mask: Optional[np.ndarray] = None,
                 params: CAParams = None):
        """
        curing, lai, slope, aspect: same-shape 2D arrays (already aligned/resampled
        to common grid - caller's responsibility, e.g. via rasterio reproject_match).
        fuel_mask: optional boolean array - False = non-flammable (water, rock, urban) -> hamesha unburnable.
        """
        self.shape = curing.shape
        self.curing = np.clip(curing, 0, 1)
        self.lai_norm = self._normalize(lai)
        self.slope = slope  # degrees
        self.aspect = aspect  # degrees, 0=N
        self.fuel_mask = fuel_mask if fuel_mask is not None else np.ones(self.shape, dtype=bool)
        self.params = params or CAParams()

    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        arr = np.nan_to_num(arr, nan=0.0)
        lo, hi = np.percentile(arr, [2, 98])
        if hi <= lo:
            return np.zeros_like(arr)
        return np.clip((arr - lo) / (hi - lo), 0, 1)

    def init_state(self, ignition_cells: List[tuple]) -> GridState:
        state = np.full(self.shape, UNBURNED, dtype=np.uint8)
        burning_since = np.zeros(self.shape, dtype=np.int32)
        for (r, c) in ignition_cells:
            if 0 <= r < self.shape[0] and 0 <= c < self.shape[1]:
                state[r, c] = BURNING
        return GridState(state=state, burning_since=burning_since)

    def _wind_alignment(self, neighbor_bearing: float, wind_dir_from: np.ndarray) -> np.ndarray:
        """
        wind_dir_from: meteorological convention, direction wind FROM aage
        (Open-Meteo 'wind_direction_10m' isi convention me hoti hai).
        Spread direction (fire jaha jaayegi) = wind_dir_from + 180.
        Alignment = cos(angle between neighbor_bearing aur spread_direction),
        clipped >=0 aur phir power se sharpen kiya (zyada downwind = zyada favor).
        """
        spread_dir = (wind_dir_from + 180.0) % 360.0
        diff = np.deg2rad(neighbor_bearing - spread_dir)
        cos_align = np.clip(np.cos(diff), 0, 1)
        return cos_align ** self.params.wind_alignment_power

    def step(self, gs: GridState, wind_speed: np.ndarray, wind_dir_from: np.ndarray) -> GridState:
        """
        Ek CA timestep aage badhao.
        wind_speed, wind_dir_from: same-shape arrays as grid (per-cell broadcast
        from per-block weather - caller resample karke deta hai).
        """
        p = self.params
        state = gs.state
        new_state = state.copy()
        burning_since = gs.burning_since.copy()

        burning_mask = state == BURNING
        unburned_mask = (state == UNBURNED) & self.fuel_mask

        # spread probability accumulate karo har unburned cell ke liye, based on jalte neighbors
        spread_prob = np.zeros(self.shape, dtype=np.float64)

        rows, cols = self.shape
        for dr, dc, bearing in NEIGHBOR_OFFSETS:
            # shift burning_mask taaki har cell ko pata chale "uske is direction wale neighbor jal raha hai kya"
            shifted = np.zeros_like(burning_mask)
            r_src_lo, r_src_hi = max(0, -dr), rows - max(0, dr)
            c_src_lo, c_src_hi = max(0, -dc), cols - max(0, dc)
            r_dst_lo, r_dst_hi = max(0, dr), rows - max(0, -dr)
            c_dst_lo, c_dst_hi = max(0, dc), cols - max(0, -dc)
            shifted[r_dst_lo:r_dst_hi, c_dst_lo:c_dst_hi] = burning_mask[r_src_lo:r_src_hi, c_src_lo:c_src_hi]

            wind_align = self._wind_alignment(bearing, wind_dir_from)
            wind_factor = 1.0 + p.wind_weight * (wind_speed / 10.0) * wind_align  # wind_speed m/s assume, /10 normalize

            neighbor_contrib = shifted.astype(np.float64) * p.base_spread_rate * wind_factor
            spread_prob += neighbor_contrib

        dryness_factor = 1.0 + p.dryness_weight * self.curing
        lai_factor = 1.0 + p.lai_weight * self.lai_norm
        slope_factor = 1.0 + p.slope_weight * np.clip(self.slope / 45.0, 0, 1)

        spread_prob = spread_prob * dryness_factor * lai_factor * slope_factor
        spread_prob = np.clip(spread_prob, 0, 1)

        # stochastic ignition: random draw har unburned cell ke liye
        rand = np.random.random(self.shape)
        ignite_now = unburned_mask & (rand < spread_prob)
        new_state[ignite_now] = BURNING

        # burning -> burnt transition (fixed duration ke baad)
        burning_since[burning_mask] += 1
        burn_out = burning_mask & (burning_since >= p.burn_duration_steps)
        new_state[burn_out] = BURNT

        return GridState(state=new_state, burning_since=burning_since)

    def run(self, gs: GridState, n_steps: int, wind_series: List[Dict]) -> List[GridState]:
        """
        wind_series: list of {"speed": array, "direction": array}, length >= n_steps
        (ek entry per timestep - hindcast/nowcast me historical hourly values,
        forecast me forecast hourly values).
        Returns list of GridState snapshots, ek per step (animation ke liye).
        """
        history = [gs]
        current = gs
        for t in range(n_steps):
            w = wind_series[min(t, len(wind_series) - 1)]
            current = self.step(current, w["speed"], w["direction"])
            history.append(current)
        return history
