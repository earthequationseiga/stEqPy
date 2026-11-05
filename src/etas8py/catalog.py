"""The data: events (Catalog) and where/when they are scored (Region)."""
import warnings
from dataclasses import dataclass

import numpy as np

from . import geo


@dataclass
class Catalog:
    """Time-sorted events in model coordinates (see geo: x, y in degrees of arc);
    t = days since the catalog start; m = magnitude minus the cutoff (so m >= 0)."""
    x: np.ndarray
    y: np.ndarray
    t: np.ndarray
    m: np.ndarray

    def __post_init__(self):
        n = len(self.t)
        if not all(len(a) == n for a in (self.x, self.y, self.m)):
            raise ValueError("x, y, t, m must have the same length")
        if np.any(np.diff(self.t) < 0):
            raise ValueError("events must be sorted by time")

    def __len__(self):
        return len(self.t)


@dataclass
class Region:
    """Study region. polygon_x/_y: vertices in model coordinates (checked, closed and made
    counter-clockwise on construction). Events inside the polygon with
    t_start <= t <= t_end are the *targets* whose likelihood is fitted; earlier/outside
    events still trigger (they enter the intensity)."""
    polygon_x: np.ndarray
    polygon_y: np.ndarray
    t_start: float
    t_end: float

    def __post_init__(self):
        self.polygon_x, self.polygon_y = geo.check_planar_polygon(
            self.polygon_x, self.polygon_y, lambda m: warnings.warn(m.strip()))
        if not self.t_start < self.t_end:
            raise ValueError("need t_start < t_end")

    @property
    def npoly(self):
        return len(self.polygon_x) - 1

    def target_flags(self, cat):
        """1 = target, 0 = on the boundary / outside the window, -1 = outside the polygon
        (the original's `ind` convention; only 1 counts)."""
        ind = geo.polyse(self.polygon_x, self.polygon_y, cat.x, cat.y)
        ind[(cat.t > self.t_end) | (cat.t < self.t_start)] = 0
        return ind
