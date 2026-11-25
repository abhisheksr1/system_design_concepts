"""
Consistent hashing + a tiny shard router, to show what sharding looks like in code.

Run:  python sharding_demo_v2.py

Part 1 is a small, traced example with hand-picked positions so every
number is easy to follow.
Part 2 is a benchmark with 20,000 keys using real MD5 hashes.

Each "shard" is a dict here. In a real system it would be a database
connection, but the routing logic is the same.
"""
import bisect
import hashlib


def stable_hash(key: str) -> int:
    # Python's built-in hash() is randomized per process, so two app servers
    # would disagree on where a key lives. MD5 gives the same number everywhere.
    return int(hashlib.md5(key.encode()).hexdigest(), 16)


class HashRing:
    """Maps keys to shard names.

    Each shard places `vnodes` markers ("virtual nodes") on a circle.
    A key belongs to the first marker clockwise from its hash position.
    """

    def __init__(self, vnodes: int = 100, hash_fn=stable_hash):
        self.vnodes = vnodes
        self.hash_fn = hash_fn
        self._points: list[int] = []      # all marker positions, kept sorted
        self._owner: dict[int, str] = {}  # marker position -> shard name

    def add_shard(self, name: str) -> None:
        for i in range(self.vnodes):
            pos = self.hash_fn(f"{name}#{i}")
            bisect.insort(self._points, pos)  # insert and keep the list sorted
            self._owner[pos] = name

    def lookup(self, key: str) -> str:
        h = self.hash_fn(key)
        # bisect returns the index of the first marker past h (clockwise).
        idx = bisect.bisect(self._points, h)
        # If h is past the last marker, idx == len(points). Wrap to index 0,
        # because the circle loops back to the start.
        idx %= len(self._points)
        return self._owner[self._points[idx]]


class ShardRouter:
    """What your app talks to. Plays the role mongos or VTGate plays:
    pick the shard from the key, then forward the call there."""

    def __init__(self, shard_names: list[str], vnodes: int = 100, hash_fn=stable_hash):
        self.ring = HashRing(vnodes, hash_fn)
        self.shards: dict[str, dict] = {}
        for name in shard_names:
            self.shards[name] = {}
            self.ring.add_shard(name)

    def put(self, key: str, value) -> None:
        self.shards[self.ring.lookup(key)][key] = value

    def get(self, key: str):
        return self.shards[self.ring.lookup(key)].get(key)

    def add_shard(self, name: str) -> int:
        """Add an empty shard, then move only the keys whose owner changed.
        Returns how many keys moved.

        This re-checks every key for clarity. A real system would copy only
        the affected key ranges, in the background.
        """
        self.shards[name] = {}
        self.ring.add_shard(name)
        moved = 0
        for shard_name, store in list(self.shards.items()):
            for key in list(store.keys()):
                new_owner = self.ring.lookup(key)
                if new_owner != shard_name:  # owner changed, so move it
                    self.shards[new_owner][key] = store.pop(key)
                    moved += 1
        return moved

    def search_all_shards(self, predicate) -> list:
        """A query without the shard key can't be routed, so it must check
        every shard and combine the results (scatter-gather)."""
        results = []
        for store in self.shards.values():
            results.extend(k for k in store if predicate(k))
        return results

    def distribution(self) -> dict[str, int]:
        return {name: len(store) for name, store in sorted(self.shards.items())}


def naive_mod_moved(keys: list[str], old_n: int, new_n: int) -> int:
    """How many keys change shard if you use hash(key) % N and bump N."""
    moved = 0
    for k in keys:
        h = stable_hash(k)  # compute once per key
        if h % old_n != h % new_n:
            moved += 1
    return moved


# ---------------------------------------------------------------------------
# Part 1: small traced example with hand-picked positions
# ---------------------------------------------------------------------------

# Positions on a ring of size 100, chosen so every step is easy to check.
POSITIONS = {
    "A#0": 10, "A#1": 60,
    "B#0": 35, "B#1": 85,
    "C#0": 20, "C#1": 70,
    "a": 5, "b": 25, "c": 50, "d": 75, "e": 95, "x": 15,
}


def readable_hash(s: str) -> int:
    return POSITIONS[s]


def print_ring(router: ShardRouter) -> None:
    ring = router.ring
    print("  ring markers (sorted):", ring._points)
    print("  owner of each marker :", {p: ring._owner[p] for p in ring._points})


def print_shards(router: ShardRouter) -> None:
    for name, store in router.shards.items():
        print(f"  shard {name}: {sorted(store)}")


def small_demo() -> None:
    print("=== Part 1: small traced example ===\n")

    router = ShardRouter(["A", "B"], vnodes=2, hash_fn=readable_hash)
    print("Step 1: add shards A and B (2 markers each)")
    print_ring(router)

    print("\nStep 2: route keys to shards (put)")
    for key in ["a", "b", "c", "d", "e", "x"]:
        h = readable_hash(key)
        owner = router.ring.lookup(key)
        print(f"  key {key!r}: position {h:>2} -> shard {owner}")
        router.put(key, f"val-{key}")
    print_shards(router)

    print("\nStep 3: add shard C (2 markers)")
    moved = router.add_shard("C")
    print_ring(router)
    print_shards(router)
    print(f"  keys moved: {moved}")
    print(f"  get('x') -> {router.get('x')!r} (now on shard {router.ring.lookup('x')})")


# ---------------------------------------------------------------------------
# Part 2: benchmark with real MD5 hashes
# ---------------------------------------------------------------------------

def benchmark() -> None:
    n_keys = 20_000
    keys = [f"user:{i}" for i in range(n_keys)]

    router = ShardRouter([f"shard-{i}" for i in range(4)])
    for k in keys:
        router.put(k, {"id": k})

    print("== 4 shards, 20,000 keys ==")
    for name, count in router.distribution().items():
        print(f"  {name}: {count:>5} keys ({count / n_keys:.1%})")

    print("\n== Naive hash(key) % N: going from 4 -> 5 shards ==")
    naive = naive_mod_moved(keys, 4, 5)
    print(f"  keys that must move: {naive} ({naive / n_keys:.1%})")

    print("\n== Consistent hashing: going from 4 -> 5 shards ==")
    moved = router.add_shard("shard-4")
    print(f"  keys that must move: {moved} ({moved / n_keys:.1%})")
    for name, count in router.distribution().items():
        print(f"  {name}: {count:>5} keys ({count / n_keys:.1%})")

    assert all(router.get(k) == {"id": k} for k in keys)
    print("\nAll keys still readable after rebalancing.")

    # A query on a non-shard-key field must check every shard.
    hits = router.search_all_shards(lambda k: k.endswith("7"))
    print(f"Search over {len(router.shards)} shards for keys ending in '7': {len(hits)} hits")


if __name__ == "__main__":
    small_demo()
    print("\n" + "=" * 60 + "\n")
    benchmark()
