import math
import cmath

class TunnelingHash:
    def __init__(self, seed: int):
        self.seed = seed
    
    def wave_function(self, x: float) -> complex:
        """Örnek dalga fonksiyonu (analojik)."""
        return cmath.exp(1j * self.seed * x)
    
    def potential_well(self, value: complex) -> complex:
        """Potansiyel kuyu dönüşümü."""
        return value * complex(math.sin(abs(value.real)), math.cos(abs(value.imag)))
    
    def hash(self, message: str) -> str:
        """Mesajı hash'le."""
        state = sum(ord(c) for c in message) + self.seed
        wf = self.wave_function(state)
        tunneled = self.potential_well(wf)
        # basit gösterim: gerçek ve imajiner kısımdan çıktı türet
        return f"{abs(tunneled.real):.16f}{abs(tunneled.imag):.16f}"
