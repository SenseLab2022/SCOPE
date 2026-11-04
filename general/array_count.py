from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import pdb

@dataclass
class Placement:
    array_id: int
    top: int
    left: int
    h: int
    w: int
    layer_idx: int

class ArrayPacker:
    def __init__(self, array_h: int, array_w: int, allow_sort:bool = False):
        self.H = array_h
        self.W = array_w
        self.allow_sort = allow_sort
        self.arrays: List[List[List[bool]]] = []  # each array is a 2D grid of bools
        self.placements: List[Placement] = []

    def _new_array(self):
        self.arrays.append([[False]*self.W for _ in range(self.H)])

    def _try_place_in_array(self, arr_idx: int, h: int, w: int) -> Optional[Tuple[int,int]]:
        occ = self.arrays[arr_idx]
        for r in range(self.H - h + 1):
            for c in range(self.W - w + 1):
                # check if can place at (r,c)
                ok = True
                for rr in range(r, r + h):
                    row = occ[rr] 
                    if any(row[cc] for cc in range(c, c + w)):
                        ok = False
                        break
                if ok:
                    # place it
                    for rr in range(r, r + h):
                        for cc in range(c, c + w):
                            occ[rr][cc] = True
                    return (r, c)
        return None

    def _place_one(self, layer_idx: int, h: int, w: int) -> int:
        # check existing arrays
        for i in range(len(self.arrays) - 1, -1, -1): #descending order
        # for i in range(len(self.arrays)):
            pos = self._try_place_in_array(i, h, w) 
            if pos is not None:
                r, c = pos
                self.placements.append(Placement(i, r, c, h, w, layer_idx))
                return i
            
        # if not placed, create new array
        self._new_array()
        i = len(self.arrays) - 1
        pos = self._try_place_in_array(i, h, w)
        assert pos is not None
        r, c = pos
        self.placements.append(Placement(i, r, c, h, w, layer_idx))
        return i

    def pack(self, layer_rects: List[Tuple[int,int]]) -> int:
        for (h, w) in layer_rects:
            if not ((1 <= h <= self.H and 1 <= w <= self.W) or
                    (self.allow_rotate and 1 <= w <= self.H and 1 <= h <= self.W)):
                raise ValueError(f"单层bank({h}x{w}) 比阵列({self.H}x{self.W})还大，无法放置")

        if self.allow_sort: 
        #sort by area desc, to minimize array count
            order = sorted(
                enumerate(layer_rects),
                key=lambda x: (x[1][0]*x[1][1], max(x[1]), min(x[1])),
                reverse=True
            )
        else:
            order = list(enumerate(layer_rects))

        for layer_idx, (h, w) in order:
            self._place_one(layer_idx, h, w)

        return len(self.arrays)

    # === ACSII Visualize ===
    def ascii_map(self, array_id: int, label_mode: str = "layer") -> str:
        grid = [["." for _ in range(self.W)] for _ in range(self.H)]
        # 取该 array 的放置
        arr_places = [p for p in self.placements if p.array_id == array_id]
        for bid, p in enumerate(arr_places):
            tag = f"L{p.layer_idx}" if label_mode == "layer" else str(bid)
            for r in range(p.top, p.top + p.h):
                for c in range(p.left, p.left + p.w):
                    grid[r][c] = tag
        # 渲染
        lines = []
        border = "+" + ("----" * self.W) + "+"
        lines.append(border)
        for r in range(self.H):
            row = "|"
            for c in range(self.W):
                cell = grid[r][c]
                row += f"{cell:>4}"
            row += " |"
            lines.append(row)
        lines.append(border)
        return "\n".join(lines)

def minimal_arrays(array_shape: Tuple[int,int], layer_rects: List[Tuple[int,int]], allow_sorted: bool= False, Hw_config = None):
    packer = ArrayPacker(array_shape[0], array_shape[1], allow_sort=allow_sorted)
    n_arrays = packer.pack(layer_rects)
    # 返回阵列数量与每个层的安放位置（按原始层顺序）
    placements_by_layer = sorted(packer.placements, key=lambda p: p.layer_idx)

    print("--------------------------Array Count----------------------------")
    print("Minimum Arrays：", n_arrays)

    for aid in range(n_arrays):
        print(f"\n----- Array #{aid} Placemant -----")
        print(packer.ascii_map(aid, label_mode="layer"))

    array = int(Hw_config.get('Architecture level', 'array_num'))
    if array < n_arrays:
        raise ValueError(f"Array_num{array} not enough, should be at least {n_arrays} arrays")


    return n_arrays, placements_by_layer

# ---------------- Example ----------------
if __name__ == "__main__":
    # Array size 4x4，Each layer 2x4、1x2、2x4
    array_shape = [4, 4]
    layers = [[2,4], [1,2], [2,4],[2,2],[2,4],[1,1],[2,4],[1,1],[4,4],[1,4],[2,1]]
    n, places = minimal_arrays(array_shape, layers,  allow_sorted=True)
    for p in places:
        print(f"Layer{p.layer_idx}: place at array{p.array_id}  (row={p.top}, col={p.left}), size{p.h}x{p.w}")
