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
  """Closed loop divided highway: straights joined by constant radius left turns, sampled every `step` meters"""
  def __init__(self, straight_a: float = 1200., straight_b: float = 600., radius: float = 350., step: float = 1.0):
    self.radius = radius
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
    self.ds = step
    self.x, self.z, self.psi, self.k = (np.array(a) for a in (xs, zs, psis, ks))
    self.length = float(sum(length for length, _ in segs))
    self.n = len(self.x) - 1  # last sample == first sample (closed loop)

  def pose(self, s: float, d: float = 0.0) -> tuple[float, float, float]:
    """World (x, z, heading) at arc length s along the centerline, offset d meters to the right"""
    s %= self.length
    f = s / self.length * self.n
    i = min(int(f), self.n - 1)
    t = f - i
    x = self.x[i] + t * (self.x[i + 1] - self.x[i])
    z = self.z[i] + t * (self.z[i + 1] - self.z[i])
    psi = self.psi[i] + t * (self.psi[i + 1] - self.psi[i])
    rx, rz = right_of(psi)
    return float(x + d * rx), float(z + d * rz), float(psi)

  def curvature(self, s: float) -> float:
    return float(self.k[int((s % self.length) / self.length * self.n)])

  def polyline(self, d: float = 0.0, y: float = 0.0, spacing: float = 4.0) -> list[list[float]]:
    """Points along the loop offset d to the right, straights are decimated since they don't need samples"""
    pts = []
    s = 0.0
    while s <= self.length + 1e-6:
      x, z, _ = self.pose(s, d)
      pts.append([round(x, 3), y, round(z, 3)])
      s += spacing if self.curvature(s) else min(50.0, spacing * 10)
    return pts

  def project(self, x: float, z: float, s_guess: float, window: float = 60.0) -> tuple[float, float]:
    """(s, d) of a world point near s_guess"""
    s = np.arange(s_guess - window, s_guess + window, self.ds)
    idx = ((s % self.length) / self.length * self.n).astype(int)
    dist2 = (self.x[idx] - x) ** 2 + (self.z[idx] - z) ** 2
    j = int(np.argmin(dist2))
    i = idx[j]
    psi = self.psi[i]
    fx, fz = math.sin(psi), math.cos(psi)
    rx, rz = right_of(psi)
    dx, dz = x - self.x[i], z - self.z[i]
    return float((s[j] + dx * fx + dz * fz) % self.length), float(dx * rx + dz * rz)
