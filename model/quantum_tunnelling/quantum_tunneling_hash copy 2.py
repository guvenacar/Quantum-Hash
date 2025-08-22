#model/quantum_tunnelling/quantum_tunneling_hash.py
import os 
import numpy as np
import matplotlib.pyplot as plt
from model.dynamic_seed_generator import polynomial_calculate, bits_right_pad


UINT64_MAX = 2**64 - 1

def _u64s_from_bytes(b: bytes):
    """64 baytı 8'lik parçalara bölüp uint64 listesine çevir."""
    if len(b) != 64:
        raise ValueError("seed_bytes tam olarak 64 bayt olmalı")
    return [int.from_bytes(b[i:i+8], "big") for i in range(0, 64, 8)]

def _map_u64(u, lo, hi):
    """uint64 -> [lo, hi] aralığına düzgün ölçekleme"""
    return lo + (u / UINT64_MAX) * (hi - lo)


class TunnelingHash:
    """
    - Bariyer: merkez + genişlik ile tanımlanır.
    - 64 baytlık seed'den (veya int'ten) dalga paketi & bariyer parametreleri türetilir.
    - Zaman evrimi: split-operator (FFT) + absorbing boundary.
    """
    def __init__(self, seed_bytes: bytes | None = None, seed_int: int | None = None,
                 N: int = 2048, L: float = 20.0):
        if seed_bytes is None and seed_int is None:
            raise ValueError("seed_bytes veya seed_int vermelisiniz")

        if seed_bytes is None:
            # tamsayıyı 64 bayta genişlet
            seed_bytes = seed_int.to_bytes(64, "big", signed=False)
        elif len(seed_bytes) != 64:
            raise ValueError("seed_bytes tam 64 bayt olmalı")

        self.seed_bytes = seed_bytes
        self.u = _u64s_from_bytes(seed_bytes)  # 8 adet uint64

        self.N = N
        self.L = L
        self.x = np.linspace(-L/2, L/2, N)
        self.dx = self.x[1] - self.x[0]
        self.hbar = 1.0
        self.m = 1.0

        self.psi = None
        self.V = None


    def params_from_seed(self):
        u = self.u
        # Daha geniş ve optimize edilmiş parametre aralıkları
        x0   = _map_u64(u[0], -8.0, 8.0)     # Daha geniş pozisyon aralığı
        k0   = _map_u64(u[1],  1.0,  15.0)   # Daha geniş momentum aralığı
        sigma = _map_u64(u[2], 0.3,  2.0)    # Daha geniş sigma
        
        center = _map_u64(u[3], -3.0, 3.0)   # Bariyer merkezi için geniş aralık
        width  = _map_u64(u[4],  0.3,  4.0)  # Bariyer genişliği
        V0     = _map_u64(u[5],  1.0, 25.0)  # Bariyer yüksekliği
        
        absorb_strength = _map_u64(u[6], 0.1, 0.5)
        dt = _map_u64(u[7], 0.001, 0.02)     # Daha geniş zaman adımı
        
        return {
            "x0": x0, "k0": k0, "sigma": sigma,
            "barrier_center": center, "barrier_width": width, "V0": V0,
            "absorb_strength": absorb_strength, "dt": dt
        }

    # --------- Dalga paketi ve potansiyel ---------
    def create_wave_packet(self, x0: float, k0: float, sigma: float):
        psi0 = np.exp(- (self.x - x0)**2 / (2 * sigma**2)) * np.exp(1j * k0 * self.x)
        psi0 /= np.sqrt(np.sum(np.abs(psi0)**2) * self.dx)  # normalize
        self.psi = psi0
        return psi0

    def define_barrier_center_width(self, center: float, width: float, V0: float):
        a = center - width/2
        b = center + width/2
        V = np.zeros_like(self.x)
        V[(self.x >= a) & (self.x <= b)] = V0
        self.V = V
        return V, (a, b)

    # --------- Absorbing boundary (kenar sönümleme) ---------
    def absorbing_boundary(self, strength: float = 0.25):
        mask = np.ones_like(self.x)
        edge = int(0.1 * self.N)  # %10'luk kenar
        ramp = np.linspace(0, 1, edge)
        mask[:edge] *= np.exp(-strength * ramp**2)
        mask[-edge:] *= np.exp(-strength * ramp[::-1]**2)
        return mask

    def evolve_tdse(self, psi, V, dt: float, Nt: int = 800, 
                    absorb_strength: float = 0.25, 
                    do_plot: bool = False):
        # Daha fazla zaman adımı ve daha yüksek çözünürlük
        if Nt < 1000:  # Minimum zaman adımı
            Nt = 1000
        
        # Daha kararlı zaman evrimi
        k = 2 * np.pi * np.fft.fftfreq(self.N, d=self.dx)
        T = np.exp(-1j * (self.hbar * k**2) / (2 * self.m) * dt / 2)
        Vop = np.exp(-1j * V * dt / self.hbar)
        mask = self.absorbing_boundary(absorb_strength)
        
        # Daha kararlı evrim
        for n in range(Nt):
            psi *= mask  # Kenar sönümleme
            psi_k = np.fft.fft(psi)
            psi = np.fft.ifft(T * psi_k)
            psi = Vop * psi
            psi_k = np.fft.fft(psi)
            psi = np.fft.ifft(T * psi_k)
        
        self.psi = psi
        return psi

    def get_hash(self, output_bits: int = 256) -> bytes:
        if self.psi is None:
            raise ValueError("Önce simülasyonu çalıştırmalısınız (evolve_tdse).")
        
        prob_density = np.abs(self.psi)**2
        
        # Normalizasyonu kontrol et ve düzelt
        total_prob = np.sum(prob_density) * self.dx
        if abs(total_prob - 1.0) > 0.01:  # %1'den fazla sapma
            prob_density /= total_prob  # Yeniden normalizasyon
        
        # Daha iyi bir ölçekleme yöntemi
        num_bytes = output_bits // 8
        compressed_data = bytearray()
        
        # Tüm olasılık dağılımını kullanarak hash üret
        for i in range(num_bytes):
            # Daha karmaşık bir karıştırma fonksiyonu
            start_idx = (i * 17) % len(prob_density)  # Asal sayı ile kaydırma
            segment = prob_density[start_idx:start_idx + 32]  # Küçük segmentler
            
            # Çoklu istatistiksel özellikleri birleştir
            mean_val = np.mean(segment)
            std_val = np.std(segment) if len(segment) > 1 else 0
            max_val = np.max(segment)
            
            # Karmaşık bir birleştirme
            combined = (int(mean_val * 1e6) ^ int(std_val * 1e4) ^ int(max_val * 1e2)) % 256
            compressed_data.append(combined)
        
        return bytes(compressed_data)           

# --------- Yardımcılar: metinden seed ---------
def text_to_bits(text: str, encoding="utf-8") -> str:
    return ''.join(f"{byte:08b}" for byte in text.encode(encoding))

def seed_from_text(message: str) -> int:
    bit_str = text_to_bits(message)
    bit_str = bits_right_pad(bit_str, 512)
    return polynomial_calculate(bit_str)

def test_quantum_hash_quality():
    """Kuantum hash kalitesini test et (ASCII formatında)"""
    test_results = []
    for i in range(100):  # Daha küçük test seti
        message = f"Test #{i}"
        seed_int = seed_from_text(message)
        
        th = TunnelingHash(seed_int=seed_int)
        P = th.params_from_seed()
        psi0 = th.create_wave_packet(P["x0"], P["k0"], P["sigma"])
        V, _ = th.define_barrier_center_width(P["barrier_center"], P["barrier_width"], P["V0"])
        
        th.evolve_tdse(psi0, V, P["dt"], Nt=1200, 
                      absorb_strength=P["absorb_strength"])
        
        hash_bytes = th.get_hash(512)
        test_results.append(hash_bytes)
    
    # NIST testi için ASCII formatında dosyaya yaz
    with open("test_quantum_hash_ascii.txt", "w") as f:
        for result in test_results:
            # Binary'i ASCII bit string'ine dönüştür (512 bit)
            binary_string = ''.join(format(byte, '08b') for byte in result)
            f.write(binary_string + "\n")
    
    print("ASCII test dosyası oluşturuldu: test_quantum_hash_ascii.txt")
    print("NIST testi için kullanılabilir.")

# --- Yeniden düzenlenmiş Ana Bölüm ---
if __name__ == "__main__":
    
    # Önce küçük test yapmak için
    print("Önce küçük test yapılıyor...")
    test_quantum_hash_quality()
    
    # Sonra tam ölçekli üretim
    # Dosyaları başlatmak için silme
    if os.path.exists("results/nist_seed_data.txt"):
        os.remove("results/nist_seed_data.txt")
    if os.path.exists("results/nist_hash_data.txt"):
        os.remove("results/nist_hash_data.txt")
        
    print("Creating seed and hash files (ASCII bitstreams)...")

    # 0'dan 999'a kadar döngü
    for i in range(1000):
        # Her döngüde farklı bir girdi kullan
        message = f"Quantum Hash Test #{i:04d}"
        seed_int = seed_from_text(message)

        th = TunnelingHash(seed_int=seed_int, N=2048, L=20.0)
        P = th.params_from_seed()
        psi0 = th.create_wave_packet(P["x0"], P["k0"], P["sigma"])
        V, _ = th.define_barrier_center_width(P["barrier_center"], P["barrier_width"], P["V0"])

        # Simülasyonu çalıştır
        th.evolve_tdse(
            psi0, V,
            dt=P["dt"], Nt=900,
            absorb_strength=P["absorb_strength"],
            do_plot=False
        )

        # Hash değerini 512 bit olarak üret
        quantum_bytes = th.get_hash(output_bits=512)

        # 1. seed_int'i 512-bit ikili stringe dönüştür
        seed_binary_string = format(seed_int, '0512b')

        # 2. quantum_bytes'ı ikili stringe dönüştür
        quantum_binary_string = ''.join(format(byte, '08b') for byte in quantum_bytes)

        # 3. İkili stringleri dosyalara ekle
        with open("results/nist_seed_data.txt", "a") as sf, \
            open("results/nist_hash_data.txt", "a") as hf:
            sf.write(seed_binary_string + "\n")  # Satır sonu ekle
            hf.write(quantum_binary_string + "\n")  # Satır sonu ekle

    print("Done. 1000 bitstream sets written to results/ folder.")