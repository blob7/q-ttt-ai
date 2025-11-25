import hashlib
import math
import uuid
from dataclasses import dataclass
from multiprocessing import get_context, shared_memory
from typing import Dict, Iterable, List, Optional

import numpy as np

BOARD_CELLS = 9 * 9


def _hash_state(board_bytes: bytes, player: int, move: int) -> tuple[int, int]:
    digest = hashlib.blake2b(
        board_bytes + bytes([(player + 256) % 256]) + move.to_bytes(2, "little", signed=True),
        digest_size=16,
    ).digest()
    lo = int.from_bytes(digest[:8], "little", signed=False)
    hi = int.from_bytes(digest[8:], "little", signed=False)
    return lo, hi


def _decode_action_key(action_key: int) -> tuple[int, int]:
    row = (action_key >> 8) & 0xFF
    col = action_key & 0xFF
    return row, col


@dataclass
class SharedSegment:
    name: str
    shape: tuple[int, ...]
    dtype: np.dtype
    init: Optional[int | float] = None


class SharedActionValueBackend:
    """Shared-memory Q/visit table with open addressing per state and fixed action slots."""

    def __init__(
        self,
        *,
        capacity: int,
        max_actions: int,
        lock_count: int,
        context: Optional[Dict] = None,
        create: bool = False,
    ) -> None:
        if context is None:
            context = {}
        self.capacity = capacity
        self.max_actions = max_actions
        self.lock_count = lock_count
        self._locks = context.get("locks")
        ctx = get_context("spawn")
        if self._locks is None:
            self._locks = [ctx.Lock() for _ in range(lock_count)]
            context["locks"] = self._locks

        if create:
            prefix = context.get("prefix") or f"sqtab_{uuid.uuid4().hex}"
            context["prefix"] = prefix
            segments = {
                "used": SharedSegment(f"{prefix}_used", (capacity,), np.uint8, 0),
                "hash_lo": SharedSegment(f"{prefix}_hlo", (capacity,), np.uint64, 0),
                "hash_hi": SharedSegment(f"{prefix}_hhi", (capacity,), np.uint64, 0),
                "player": SharedSegment(f"{prefix}_ply", (capacity,), np.int8, 0),
                "last_move": SharedSegment(f"{prefix}_lmo", (capacity,), np.int32, -1),
                "action_count": SharedSegment(f"{prefix}_act", (capacity,), np.int16, 0),
                "canonical": SharedSegment(f"{prefix}_can", (capacity, BOARD_CELLS), np.int8, 0),
                "action_keys": SharedSegment(f"{prefix}_akey", (capacity, max_actions), np.int16, -1),
                "q_values": SharedSegment(f"{prefix}_qval", (capacity, max_actions), np.float16, 0.0),
                "visit_counts": SharedSegment(f"{prefix}_vct", (capacity, max_actions), np.int32, 0),
            }
            context["segments"] = {name: seg.name for name, seg in segments.items()}
            self._segments = {}
            for name, segment in segments.items():
                shm = shared_memory.SharedMemory(create=True, size=int(np.dtype(segment.dtype).itemsize * math.prod(segment.shape)), name=segment.name)
                array = np.ndarray(segment.shape, dtype=segment.dtype, buffer=shm.buf)
                if segment.init is not None:
                    array.fill(segment.init)
                self._segments[name] = (shm, array)
            context["owner"] = True
        else:
            segment_names = context["segments"]
            self._segments = {}
            for key, shm_name in segment_names.items():
                dtype, shape = self._segment_signature(key)
                shm = shared_memory.SharedMemory(name=shm_name)
                array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
                self._segments[key] = (shm, array)
            context["owner"] = False

        self._context = context
        self._owner = context.get("owner", False)

        # Fast references to arrays
        self.state_used = self._segments["used"][1]
        self.state_hash_lo = self._segments["hash_lo"][1]
        self.state_hash_hi = self._segments["hash_hi"][1]
        self.state_player = self._segments["player"][1]
        self.state_last_move = self._segments["last_move"][1]
        self.state_action_count = self._segments["action_count"][1]
        self.state_canonical = self._segments["canonical"][1]
        self.action_keys = self._segments["action_keys"][1]
        self.q_values = self._segments["q_values"][1]
        self.visit_counts = self._segments["visit_counts"][1]

    def _segment_signature(self, key: str) -> tuple[np.dtype, tuple[int, ...]]:
        if key == "used":
            return (np.uint8, (self.capacity,))
        if key in {"hash_lo", "hash_hi"}:
            return (np.uint64, (self.capacity,))
        if key == "player":
            return (np.int8, (self.capacity,))
        if key == "last_move":
            return (np.int32, (self.capacity,))
        if key == "action_count":
            return (np.int16, (self.capacity,))
        if key == "canonical":
            return (np.int8, (self.capacity, BOARD_CELLS))
        if key == "action_keys":
            return (np.int16, (self.capacity, self.max_actions))
        if key == "q_values":
            return (np.float16, (self.capacity, self.max_actions))
        if key == "visit_counts":
            return (np.int32, (self.capacity, self.max_actions))
        raise KeyError(key)

    @classmethod
    def create(
        cls,
        *,
        capacity: int = 120_000,
        max_actions: int = 81,
        lock_count: int = 2048,
    ) -> "SharedActionValueBackend":
        return cls(capacity=capacity, max_actions=max_actions, lock_count=lock_count, create=True)

    @staticmethod
    def bytes_per_state(max_actions: int) -> int:
        base = (
            np.dtype(np.uint8).itemsize  # used flag
            + np.dtype(np.uint64).itemsize * 2  # hash parts
            + np.dtype(np.int8).itemsize  # player
            + np.dtype(np.int32).itemsize  # last_move
            + np.dtype(np.int16).itemsize  # action_count
            + np.dtype(np.int8).itemsize * BOARD_CELLS  # canonical board
        )
        per_action = (
            np.dtype(np.int16).itemsize  # action key
            + np.dtype(np.float16).itemsize  # q value
            + np.dtype(np.int32).itemsize  # visit count
        )
        return base + per_action * max_actions

    @staticmethod
    def capacity_for_memory(memory_mb: int, max_actions: int, *, safety_margin: float = 0.9) -> int:
        if memory_mb <= 0:
            return 0
        bytes_total = int(memory_mb * 1024 * 1024 * safety_margin)
        per_state = SharedActionValueBackend.bytes_per_state(max_actions)
        if per_state <= 0:
            return 0
        return max(8, bytes_total // per_state)

    def fork(self) -> "SharedActionValueBackend":
        context = {
            "prefix": self._context["prefix"],
            "segments": self._context["segments"],
            "locks": self._context["locks"],
            "owner": False,
        }
        return SharedActionValueBackend(
            capacity=self.capacity,
            max_actions=self.max_actions,
            lock_count=self.lock_count,
            context=context,
            create=False,
        )

    def close(self, *, unlink: bool = False) -> None:
        for shm, _ in self._segments.values():
            shm.close()
            if unlink and self._owner:
                shm.unlink()

    def __getstate__(self):
        return {
            "capacity": self.capacity,
            "max_actions": self.max_actions,
            "lock_count": self.lock_count,
            "context": {
                "prefix": self._context["prefix"],
                "segments": self._context["segments"],
                "locks": self._context["locks"],
                "owner": False,
            },
        }

    def __setstate__(self, state):
        backend = SharedActionValueBackend(
            capacity=state["capacity"],
            max_actions=state["max_actions"],
            lock_count=state["lock_count"],
            context=state["context"],
            create=False,
        )
        self.__dict__.update(backend.__dict__)

    # --- Internal helpers -------------------------------------------------

    def _lock_for(self, index: int):
        return self._locks[index % self.lock_count]

    def _ensure_state(self, state: tuple[bytes, int, int]) -> int:
        board_bytes, player, move = state
        hash_lo, hash_hi = _hash_state(board_bytes, player, move)
        idx = hash_lo % self.capacity
        board_arr = np.frombuffer(board_bytes, dtype=np.int8, count=BOARD_CELLS)

        for _ in range(self.capacity):
            lock = self._lock_for(idx)
            with lock:
                if self.state_used[idx]:
                    if (
                        self.state_hash_lo[idx] == hash_lo
                        and self.state_hash_hi[idx] == hash_hi
                        and self.state_player[idx] == player
                        and self.state_last_move[idx] == move
                        and np.array_equal(self.state_canonical[idx], board_arr)
                    ):
                        return idx
                else:
                    self.state_used[idx] = 1
                    self.state_hash_lo[idx] = hash_lo
                    self.state_hash_hi[idx] = hash_hi
                    self.state_player[idx] = player
                    self.state_last_move[idx] = move
                    self.state_action_count[idx] = 0
                    self.action_keys[idx].fill(-1)
                    self.q_values[idx].fill(0.0)
                    self.visit_counts[idx].fill(0)
                    self.state_canonical[idx][:] = board_arr
                    return idx
            idx = (idx + 1) % self.capacity
        raise RuntimeError("Shared Q-table capacity exhausted")

    def _locate_state(self, state: tuple[bytes, int, int]) -> Optional[int]:
        board_bytes, player, move = state
        hash_lo, hash_hi = _hash_state(board_bytes, player, move)
        idx = hash_lo % self.capacity
        board_arr = np.frombuffer(board_bytes, dtype=np.int8, count=BOARD_CELLS)
        for _ in range(self.capacity):
            if not self.state_used[idx]:
                return None
            if (
                self.state_hash_lo[idx] == hash_lo
                and self.state_hash_hi[idx] == hash_hi
                and self.state_player[idx] == player
                and self.state_last_move[idx] == move
                and np.array_equal(self.state_canonical[idx], board_arr)
            ):
                return idx
            idx = (idx + 1) % self.capacity
        return None

    def _ensure_action_slot(self, state_idx: int, action_key: int) -> int:
        lock = self._lock_for(state_idx)
        with lock:
            count = int(self.state_action_count[state_idx])
            row_keys = self.action_keys[state_idx]
            for pos in range(count):
                if row_keys[pos] == action_key:
                    return pos

            if count >= self.max_actions:
                pos = self.max_actions - 1
            else:
                pos = count
                self.state_action_count[state_idx] = count + 1
            self.action_keys[state_idx, pos] = action_key
            return pos

    # --- Public API -------------------------------------------------------

    def get_state_q_snapshot(self, state: tuple[bytes, int, int]) -> Dict[int, float]:
        idx = self._locate_state(state)
        if idx is None:
            return {}
        count = int(self.state_action_count[idx])
        if count == 0:
            return {}
        lock = self._lock_for(idx)
        with lock:
            keys = self.action_keys[idx, :count].copy()
            values = self.q_values[idx, :count].copy()
        return {int(k): float(v) for k, v in zip(keys, values) if k >= 0}

    def increment_q_value(
        self,
        state: tuple[bytes, int, int],
        action_key: int,
        delta: float,
        *,
        rounding: Optional[int] = None,
    ) -> float:
        idx = self._ensure_state(state)
        pos = self._ensure_action_slot(idx, action_key)
        lock = self._lock_for(idx)
        with lock:
            prev = float(self.q_values[idx, pos])
            if rounding is not None:
                new_value = round(prev + delta, rounding)
                applied = new_value - prev
            else:
                new_value = prev + delta
                applied = delta
            self.q_values[idx, pos] = new_value
            return applied

    def increment_visit_count(self, state: tuple[bytes, int, int], action_key: int, increment: int) -> int:
        idx = self._ensure_state(state)
        pos = self._ensure_action_slot(idx, action_key)
        lock = self._lock_for(idx)
        with lock:
            prev = int(self.visit_counts[idx, pos])
            new_count = prev + increment
            self.visit_counts[idx, pos] = new_count
            return new_count - prev

    def set_q_value(self, state: tuple[bytes, int, int], action_key: int, value: float) -> None:
        idx = self._ensure_state(state)
        pos = self._ensure_action_slot(idx, action_key)
        lock = self._lock_for(idx)
        with lock:
            self.q_values[idx, pos] = float(value)

    def set_visit_count(self, state: tuple[bytes, int, int], action_key: int, count: int) -> None:
        idx = self._ensure_state(state)
        pos = self._ensure_action_slot(idx, action_key)
        lock = self._lock_for(idx)
        with lock:
            self.visit_counts[idx, pos] = int(count)

    def merge_q_value(self, state: tuple[bytes, int, int], action_key: int, value: float, visits: int) -> None:
        idx = self._ensure_state(state)
        pos = self._ensure_action_slot(idx, action_key)
        lock = self._lock_for(idx)
        with lock:
            current_visits = int(self.visit_counts[idx, pos])
            total = current_visits + visits
            if total <= 0:
                return
            prev_q = float(self.q_values[idx, pos])
            new_q = (prev_q * current_visits + value * visits) / total
            self.q_values[idx, pos] = new_q
            self.visit_counts[idx, pos] = total

    def estimate_memory_bytes(self) -> int:
        total = 0
        for shm, _ in self._segments.values():
            total += shm.size
        return total

    def export_tables(self) -> tuple[Dict[tuple[bytes, int, int], Dict[int, float]], Dict[tuple[bytes, int, int], Dict[int, int]]]:
        q_table: Dict[tuple[bytes, int, int], Dict[int, float]] = {}
        visits_table: Dict[tuple[bytes, int, int], Dict[int, int]] = {}
        for idx in range(self.capacity):
            if not self.state_used[idx]:
                continue
            board_bytes = self.state_canonical[idx].tobytes()
            player = int(self.state_player[idx])
            last_move = int(self.state_last_move[idx])
            key = (board_bytes, player, last_move)
            count = int(self.state_action_count[idx])
            if count <= 0:
                continue
            action_keys = self.action_keys[idx, :count]
            q_vals = self.q_values[idx, :count]
            v_counts = self.visit_counts[idx, :count]
            q_table[key] = {int(a): float(v) for a, v in zip(action_keys, q_vals) if a >= 0}
            visits_table[key] = {int(a): int(v) for a, v in zip(action_keys, v_counts) if a >= 0}
        return q_table, visits_table

    def iter_states(self) -> Iterable[tuple[int, tuple[bytes, int, int]]]:
        for idx in range(self.capacity):
            if not self.state_used[idx]:
                continue
            board_bytes = self.state_canonical[idx].tobytes()
            player = int(self.state_player[idx])
            last_move = int(self.state_last_move[idx])
            yield idx, (board_bytes, player, last_move)

    def get_state_action_data(self, idx: int) -> tuple[List[int], List[float], List[int]]:
        count = int(self.state_action_count[idx])
        if count <= 0:
            return [], [], []
        keys = self.action_keys[idx, :count].tolist()
        qvals = self.q_values[idx, :count].tolist()
        visits = self.visit_counts[idx, :count].tolist()
        return keys, qvals, visits