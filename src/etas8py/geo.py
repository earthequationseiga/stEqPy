"""Units, study-polygon input/checks and map projections.

Model coordinates are always "degrees": 1 unit = 1 degree of great-circle arc
(KM_PER_DEG km). Every projection outputs these units, so the ETAS parameter D (deg^2),
the bandwidth floor bwm (deg) and the numerical guards keep the same scale whatever the
projection, and the optimiser (not scale-invariant) sees the same conditioning.

Projections (lon/lat in degrees -> model x, y):
- "flat": the original. x = lon * cos(lat_c), y = lat, lat_c = area-centroid latitude of
  the polygon. Distance error grows with |lat - lat_c| (JMA: -6%..+6% p1-p99).
- "laea": Lambert azimuthal equal-area about the polygon's area centroid. Areas exact
  (consistent with the polygon integrals); distance error ~ (angle from centre)^2 / 8.
"""
import math

import numpy as np

EARTH_RADIUS_KM = 6371.0
KM_PER_DEG = EARTH_RADIUS_KM * math.pi / 180.0
DEG = 3.1415926535 / 180.0  # the original's degree->radian constant (flat projection)

_LENGTH = {"deg": 1.0, "km": 1.0 / KM_PER_DEG}
_AREA = {"deg2": 1.0, "km2": 1.0 / KM_PER_DEG ** 2}


def _parse(value, table, what):
    if not isinstance(value, str):
        raise ValueError(f"{what} needs units, e.g. '{value} {next(iter(table))}' "
                         f"(allowed: {', '.join(table)})")
    parts = value.split()
    if len(parts) != 2 or parts[1] not in table:
        raise ValueError(f"cannot read {what} = {value!r}; use '<number> <unit>' with unit "
                         f"in {', '.join(table)}")
    return float(parts[0]) * table[parts[1]]


def length_deg(value, what="length"):
    """'0.05 deg' / '5.56 km' -> degrees of arc."""
    return _parse(value, _LENGTH, what)


def area_deg2(value, what="area"):
    """'0.0018 deg2' / '22.7 km2' -> square degrees of arc."""
    return _parse(value, _AREA, what)


# -- polygon --------------------------------------------------------------------------------
def read_polygon(path):
    """Text file, one 'lon lat' vertex per line (space or comma separated, decimal degrees).
    '#' starts a comment; blank lines are ignored; closing the ring is optional."""
    pts = []
    for n, line in enumerate(open(path, encoding="utf-8"), 1):
        line = line.split("#")[0].replace(",", " ").split()
        if not line:
            continue
        if len(line) != 2:
            raise ValueError(f"{path}:{n}: expected 'lon lat', got {' '.join(line)!r}")
        pts.append((float(line[0]), float(line[1])))
    a = np.array(pts, dtype=np.float64).reshape(-1, 2)
    return a[:, 0], a[:, 1]


def check_lonlat_polygon(lon, lat):
    """Open (unclosed) lon/lat vertex arrays after the geographic checks."""
    lon, lat = np.asarray(lon, np.float64), np.asarray(lat, np.float64)
    if len(lon) > 1 and lon[0] == lon[-1] and lat[0] == lat[-1]:
        lon, lat = lon[:-1], lat[:-1]
    if np.any(np.abs(lat) > 90) or np.any(lon < -180) or np.any(lon > 360):
        raise ValueError("polygon: need -90 <= lat <= 90 and -180 <= lon <= 360")
    if np.ptp(lon) > 180:
        raise ValueError("polygon spans more than 180 deg of longitude: if it crosses the "
                         "antimeridian, give longitudes in 0..360 instead of -180..180")
    return lon, lat


def check_planar_polygon(x, y, log=None):
    """Closed, counter-clockwise model-coordinate polygon, or ValueError.
    The polygon integrals use Green's theorem with signed triangles: a clockwise ring
    integrates to -1 instead of +1 and silently breaks the fit, so it is reversed here."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if len(x) > 1 and x[0] == x[-1] and y[0] == y[-1]:
        x, y = x[:-1], y[:-1]
    n = len(x)
    if n < 3:
        raise ValueError("polygon needs at least 3 vertices")
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    if np.any((x == xn) & (y == yn)):
        raise ValueError("polygon has repeated consecutive vertices")
    area2 = float(np.sum(x * yn - xn * y))
    if area2 == 0.0:
        raise ValueError("polygon has zero area")
    for i in range(n):                       # O(n^2): fine for hand-made polygons
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue                     # adjacent through the closing edge
            if _segments_cross(x[i], y[i], xn[i], yn[i], x[j], y[j], xn[j], yn[j]):
                raise ValueError(f"polygon edges {i + 1} and {j + 1} intersect")
    if area2 < 0:
        if log:
            log(" polygon was clockwise: vertex order reversed")
        x, y = x[::-1].copy(), y[::-1].copy()
    return np.append(x, x[0]), np.append(y, y[0])


def _segments_cross(ax, ay, bx, by, cx, cy, dx, dy):
    """Segments AB and CD share at least one point (touching counts)."""
    def orient(px, py, qx, qy, rx, ry):
        v = (qx - px) * (ry - py) - (qy - py) * (rx - px)
        return (v > 0) - (v < 0)

    def on_seg(px, py, qx, qy, rx, ry):      # r on segment pq, given collinear
        return min(px, qx) <= rx <= max(px, qx) and min(py, qy) <= ry <= max(py, qy)

    o1, o2 = orient(ax, ay, bx, by, cx, cy), orient(ax, ay, bx, by, dx, dy)
    o3, o4 = orient(cx, cy, dx, dy, ax, ay), orient(cx, cy, dx, dy, bx, by)
    if o1 != o2 and o3 != o4:
        return True
    return ((o1 == 0 and on_seg(ax, ay, bx, by, cx, cy)) or
            (o2 == 0 and on_seg(ax, ay, bx, by, dx, dy)) or
            (o3 == 0 and on_seg(cx, cy, dx, dy, ax, ay)) or
            (o4 == 0 and on_seg(cx, cy, dx, dy, bx, by)))


def _centroid(tx, ty):
    """Area centroid of an open polygon (same loop as the original's ycentroid)."""
    n = len(tx)
    area = xm = ym = 0.0
    for i in range(n):
        i1 = (i + 1) % n
        t = tx[i] * ty[i1] - tx[i1] * ty[i]
        area += t
        xm += (tx[i] + tx[i1]) * t
        ym += (ty[i] + ty[i1]) * t
    area /= 2
    return xm / 6.0 / area, ym / 6.0 / area


# -- projections ----------------------------------------------------------------------------
class Projection:
    """lon/lat (deg) -> model x, y (deg of arc), centred on the polygon's area centroid."""

    def __init__(self, kind, poly_lon, poly_lat):
        if kind not in ("flat", "laea"):
            raise ValueError(f"projection must be 'flat' or 'laea', not {kind!r}")
        self.kind = kind
        self.lon0, self.lat0 = _centroid(np.asarray(poly_lon), np.asarray(poly_lat))
        if kind == "flat":
            from .run import ycentroid      # the original's loop, for bit-exact x scaling
            self.lat0 = ycentroid(poly_lon, poly_lat)
            self.cosm = math.cos(self.lat0 * DEG)

    def __call__(self, lon, lat):
        lon, lat = np.asarray(lon, np.float64), np.asarray(lat, np.float64)
        if self.kind == "flat":
            return lon * self.cosm, lat.copy()
        lam = np.radians(lon - self.lon0)
        phi, phi0 = np.radians(lat), math.radians(self.lat0)
        k = np.sqrt(2.0 / (1.0 + math.sin(phi0) * np.sin(phi)
                           + math.cos(phi0) * np.cos(phi) * np.cos(lam)))
        r = 180.0 / math.pi                  # sphere radius in degrees of arc
        x = r * k * np.cos(phi) * np.sin(lam)
        y = r * k * (math.cos(phi0) * np.sin(phi) - math.sin(phi0) * np.cos(phi) * np.cos(lam))
        return x, y


def great_circle_deg(lon1, lat1, lon2, lat2):
    """Haversine distance in degrees of arc."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = (np.sin((p2 - p1) / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2)
    return np.degrees(2 * np.arcsin(np.sqrt(np.minimum(a, 1.0))))
