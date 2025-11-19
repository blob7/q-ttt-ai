import psutil
from typing import Optional

def get_process() -> Optional[psutil.Process]:
    try:
        return psutil.Process()
    except Exception:
        print("psutil not available; memory_stop_threshold_mb ignored.")
        return None

def memory_limit_hit(process, threshold_mb: Optional[int]) -> bool:
    if process is None or threshold_mb is None:
        return False
    mem = process.memory_info().rss / (1024 * 1024)
    return mem >= threshold_mb
