import math
import numpy as np

# World frame (ZoomEngine): y up, ground plane is xz. A heading psi faces (sin(psi), cos(psi)) in (x, z),
# which is what an object with rotation [0, degrees(psi), 0] faces (its local +z). Increasing psi turns left.
# Lateral offset d is measured to the RIGHT of the direction of travel, right = (-cos(psi), sin(psi)).

LANE_WIDTH = 3.7
LANES = 3


def right_of(psi):
  return -np.cos(psi), np.sin(psi)


class Track:
  """Closed loop sampled every `step` meters: position, heading, curvature and road height along arc length s.
  The default is a generated highway loop, straights joined by constant radius left turns on flat ground."""
  def __init__(self, straight_a: float = 1200., straight_b: float = 600., radius: float = 350., step: float = 1.0):
    segs: list[tuple[float, float]] = []  # (length, curvature)
    for straight in (straight_a, straight_b, straight_a, straight_b):
      segs.append((straight, 0.0))
      segs.append((0.5 * math.pi * radius, 1.0 / radius))

    xs, zs, psis, ks = [0.0], [0.0], [0.0], [0.0]
    for length, k in segs:
      n = max(1, round(length / step))
      ds = length / n
      for _ in range(n):
        psi0 = psis[-1]
        psi1 = psi0 + k * ds
        mid = 0.5 * (psi0 + psi1)
        xs.append(xs[-1] + ds * math.sin(mid))
        zs.append(zs[-1] + ds * math.cos(mid))
        psis.append(psi1)
        ks.append(k)
    self._set(np.array(xs), np.array(zs), np.zeros(len(xs)), np.array(psis), np.array(ks), step,
              float(sum(length for length, _ in segs)))

  @classmethod
  def from_points(cls, pts, step: float = 1.0, closed: bool = True) -> "Track":
    """Track through ordered [x, y, z] points (y up), resampled every `step` meters. closed: a loop (the last point joins the
    first); otherwise an open route, arc lengths clamp to its ends."""
    pts = np.asarray(pts, dtype=float)[:, :3]
    if closed and np.linalg.norm(pts[0] - pts[-1]) > 1e-6:
      pts = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(pts[:, [0, 2]], axis=0), axis=1)
    keep = np.concatenate([[True], seg > 1e-3])
    pts = pts[keep]
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts[:, [0, 2]], axis=0), axis=1))])
    length = float(s[-1])
    n = max(8, round(length / step))
    ss = np.linspace(0.0, length, n + 1)
    x, y, z = (np.interp(ss, s, pts[:, i]) for i in range(3))
    # heading and curvature from the smoothed tangent
    dx, dz = np.gradient(x), np.gradient(z)
    psi = np.unwrap(np.arctan2(dx, dz))
    k = np.gradient(psi) / (length / n)
    k = np.convolve(np.pad(k, 10, mode="wrap" if closed else "edge"), np.ones(21) / 21, mode="valid")
    t = cls.__new__(cls)
    t._set(x, z, y, psi, k, length / n, length, closed)
    return t

  def _set(self, x, z, y, psi, k, ds, length, closed=True):
    self.x, self.z, self.y, self.psi, self.k = x, z, y, psi, k
    self.ds = ds
    self.length = length
    self.closed = closed
    self.n = len(self.x) - 1  # closed: last sample == first sample
    self.grade = np.gradient(y) / ds

  def _wrap(self, s):
    return s % self.length if self.closed else np.clip(s, 0.0, self.length)

  def _idx(self, s: float) -> tuple[int, float]:
    f = self._wrap(s) / self.length * self.n
    i = min(int(f), self.n - 1)
    return i, f - i

  def pose(self, s: float, d: float = 0.0) -> tuple[float, float, float]:
    """World (x, z, heading) at arc length s along the centerline, offset d meters to the right"""
    i, t = self._idx(s)
    x = self.x[i] + t * (self.x[i + 1] - self.x[i])
    z = self.z[i] + t * (self.z[i + 1] - self.z[i])
    psi = self.psi[i] + t * (self.psi[i + 1] - self.psi[i])
    rx, rz = right_of(psi)
    return float(x + d * rx), float(z + d * rz), float(psi)

  def height(self, s: float) -> float:
    i, t = self._idx(s)
    return float(self.y[i] + t * (self.y[i + 1] - self.y[i]))

  def pitch(self, s: float) -> float:
    """road pitch along the direction of travel, positive uphill [rad]"""
    return float(math.atan(self.grade[self._idx(s)[0]]))

  def curvature(self, s: float) -> float:
    return float(self.k[self._idx(s)[0]])

  def polyline(self, d: float = 0.0, y: float = 0.0, spacing: float = 4.0) -> list[list[float]]:
    """Points along the loop offset d to the right, `y` above the road. Straight flat stretches are decimated."""
    pts = []
    s = 0.0
    while s <= self.length + 1e-6:
      x, z, _ = self.pose(s, d)
      pts.append([round(x, 3), round(self.height(s) + y, 3), round(z, 3)])
      flat = abs(self.curvature(s)) < 1e-5 and abs(self.grade[self._idx(s)[0]]) < 1e-5
      s += min(50.0, spacing * 10) if flat else spacing
    return pts

  def project(self, x: float, z: float, s_guess: float, window: float = 60.0) -> tuple[float, float]:
    """(s, d) of a world point near s_guess"""
    s = self._wrap(np.arange(s_guess - window, s_guess + window, self.ds))
    idx = np.minimum((s / self.length * self.n).astype(int), self.n - 1)
    dist2 = (self.x[idx] - x) ** 2 + (self.z[idx] - z) ** 2
    j = int(np.argmin(dist2))
    i = idx[j]
    psi = self.psi[i]
    fx, fz = math.sin(psi), math.cos(psi)
    rx, rz = right_of(psi)
    dx, dz = x - self.x[i], z - self.z[i]
    return float(self._wrap(s[j] + dx * fx + dz * fz)), float(dx * rx + dz * rz)


def catmull_rom(nodes, step: float = 2.0) -> np.ndarray:
  """Centripetal Catmull-Rom spline through an open list of [x, y, z, ...] control points (BeamNG road nodes)"""
  P = np.asarray(nodes, dtype=float)[:, :3]
  if len(P) < 3:
    return P
  P = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
  out = []
  for i in range(1, len(P) - 2):
    p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
    t0 = 0.0
    t1 = t0 + max(np.linalg.norm(p1 - p0), 1e-6) ** 0.5
    t2 = t1 + max(np.linalg.norm(p2 - p1), 1e-6) ** 0.5
    t3 = t2 + max(np.linalg.norm(p3 - p2), 1e-6) ** 0.5
    n = max(2, int(np.linalg.norm(p2 - p1) / step))
    for t in np.linspace(t1, t2, n, endpoint=False):
      a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
      a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
      a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
      b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
      b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
      out.append((t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2)
  out.append(P[-2])
  return np.array(out)
