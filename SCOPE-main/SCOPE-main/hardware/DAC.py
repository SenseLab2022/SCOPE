import numpy as np
from general.param_runtime import P
import random
import pdb
from typing import List, Tuple, Optional

# ===== Tool =====
def _gauss(mu, sigma):
    return mu if sigma <= 0.0 else random.gauss(mu, sigma)

class DACBase(object):
    def __init__(self, n_bits: int, vref_h: float = 1.0, vref_l: float = 0.0):
        assert n_bits >= 1
        self.n_bits = n_bits
        self.vref_h = vref_h
        self.vref_l = vref_l

    @property
    def full_scale(self) -> float:
        return self.vref_h - self.vref_l

    def code_to_voltage(self, code: int) -> float:
        raise NotImplementedError

    def ideal_voltage(self, code: int) -> float:
        max_code = (1 << self.n_bits) - 1
        code_clamped = max(0, min(code, max_code))
        return self.vref_l + self.full_scale * code_clamped / max(1, max_code)

    def eval_dnl_inl(self) -> Tuple[List[float], List[float]]:
        N = 1 << self.n_bits
        ys = [self.code_to_voltage(c) for c in range(N)]
        y_ideal = [self.ideal_voltage(c) for c in range(N)]
        lsb = self.full_scale / (N - 1) if N > 1 else self.full_scale

        dnl = [0.0] * N
        inl = [0.0] * N
        for k in range(1, N):
            step = ys[k] - ys[k - 1]
            dnl[k] = (step / lsb) - 1.0
        dnl[0] = 0.0  

        for k in range(N):
            inl[k] = (ys[k] - y_ideal[k]) / (lsb if lsb != 0 else 1.0)

        return dnl, inl
    

        

        
