"""
A* pathfinding for the Craftax gridworld.

The map is a (num_levels, 48, 48) array of BlockType integer values.
Player positions use (x, y) where x is the row index and y is the column index.

Actions returned:
    1 = LEFT  (y - 1)
    2 = RIGHT (y + 1)
    3 = UP    (x - 1)
    4 = DOWN  (x + 1)
"""

import heapq
import numpy as np

from craftax.craftax_coop.constants import SOLID_BLOCKS


# Build a plain numpy boolean array so this module works without JAX at import time.
# Index corresponds to BlockType.value; True means the tile is non-walkable.
_MAX_BLOCK_TYPE = 64  # generous upper bound
_SOLID = np.zeros(_MAX_BLOCK_TYPE, dtype=bool)
for _v in SOLID_BLOCKS:
    _SOLID[_v] = True

# (dx, dy, action_id) — matches DIRECTIONS in constants.py
_MOVES = [
    (-1, 0, 3),  # UP
    ( 1, 0, 4),  # DOWN
    ( 0,-1, 1),  # LEFT
    ( 0, 1, 2),  # RIGHT
]


def astar(map_grid, start, goal, level=0):
    """Return a list of actions that move a player from *start* to *goal*.

    Args:
        map_grid: array-like of shape (num_levels, H, W) or (H, W) containing
                  BlockType integer values.  JAX arrays are converted to numpy
                  automatically.
        start:    (x, y) tuple for the player's current position.
        goal:     (x, y) tuple for the desired destination.
        level:    dungeon level index to search on (ignored for 2-D input).

    Returns:
        A list of integer actions [1–4] representing the shortest path, or
        None if no path exists (goal unreachable or out of bounds).
    """
    # Accept JAX arrays without importing jax at module level.
    if hasattr(map_grid, "device"):  # jax DeviceArray / ArrayImpl
        import jax
        grid = np.array(jax.device_get(map_grid))
    else:
        grid = np.asarray(map_grid)

    if grid.ndim == 3:
        grid = grid[level]

    H, W = grid.shape
    start = tuple(start)
    goal = tuple(goal)

    if start == goal:
        return []

    def is_walkable(x, y):
        return (
            0 <= x < H
            and 0 <= y < W
            and not _SOLID[grid[x, y]]
        )

    if not is_walkable(*goal):
        return None

    def h(pos):
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    # heap entries: (f, g, position, path)
    open_heap = [(h(start), 0, start, [])]
    visited = set()

    while open_heap:
        f, g, pos, path = heapq.heappop(open_heap)

        if pos in visited:
            continue
        visited.add(pos)

        if pos == goal:
            return path

        for dx, dy, action in _MOVES:
            nx, ny = pos[0] + dx, pos[1] + dy
            npos = (nx, ny)
            if npos not in visited and is_walkable(nx, ny):
                ng = g + 1
                heapq.heappush(open_heap, (ng + h(npos), ng, npos, path + [action]))

    return None  # no path found
