import json
import mmap
import os
import struct
import time
import urllib.request

import numpy as np

SHM_MAGIC = 0x5A534331  # 'ZSC1'
SHM_HEADER = 64


class ZoomAPI:
  """Client for the ZoomEngine agent API (POST JSON to http://localhost:<port>/api)"""
  def __init__(self, host: str = "127.0.0.1", port: int = 9090, timeout: float = 2.0):
    self.url = f"http://{host}:{port}/api"
    self.timeout = timeout

  def _post(self, payload: dict) -> dict:
    req = urllib.request.Request(self.url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=self.timeout) as resp:
      return json.loads(resp.read() or b"{}")

  def send(self, cmd: dict) -> dict:
    return self._post(cmd)

  def batch(self, cmds: list[dict], chunk: int = 500) -> None:
    for i in range(0, len(cmds), chunk):
      self._post({"action": "batch", "commands": cmds[i:i + chunk]})

  def query(self, what: str = "stats") -> dict:
    return self._post({"action": "query", "what": what})

  def wait_ready(self, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while True:
      try:
        self.query()
        return
      except OSError:
        if time.monotonic() > deadline:
          raise RuntimeError(f"ZoomEngine API not reachable at {self.url}, start it with `./build/ModernEngine traffic`") from None
        time.sleep(0.5)


class SensorReader:
  """Reads the latest RGB frame a ZoomEngine sensor camera writes to shared memory (seqlock protocol)"""
  def __init__(self, path: str, width: int, height: int):
    self.path = path
    self.width, self.height = width, height
    self.mm: mmap.mmap | None = None
    self.last_seq = 0

  def _open(self) -> bool:
    size = SHM_HEADER + self.width * self.height * 3
    if not os.path.exists(self.path) or os.path.getsize(self.path) < size:
      return False
    with open(self.path, "rb") as f:
      self.mm = mmap.mmap(f.fileno(), size, prot=mmap.PROT_READ)
    return True

  def read(self, out: np.ndarray) -> bool:
    """Copy the newest complete frame into out (H, W, 3). Returns False if there's no new frame."""
    if self.mm is None and not self._open():
      return False
    assert self.mm is not None
    for _ in range(10):
      magic, _version, seq0, w, h = struct.unpack_from("<IIQII", self.mm, 0)
      if magic != SHM_MAGIC or (w, h) != (self.width, self.height):
        return False
      if seq0 & 1:
        time.sleep(0.0005)
        continue
      if seq0 == self.last_seq:
        return False
      out[:] = np.frombuffer(self.mm, dtype=np.uint8, count=w * h * 3, offset=SHM_HEADER).reshape(h, w, 3)
      seq1 = struct.unpack_from("<Q", self.mm, 8)[0]
      if seq1 == seq0:
        self.last_seq = seq0
        return True
    return False

  def close(self) -> None:
    if self.mm is not None:
      self.mm.close()
      self.mm = None
