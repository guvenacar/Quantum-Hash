# model/quantum_tunnelling/quantum_tunneling_hash.py
import os
import numpy as np
import matplotlib.pyplot as plt
from model.dynamic_seed_generator import polynomial_calculate, bits_right_pad

# Taşma uyarılarını sessize al (biz mod 2^64 wrap-around istiyoruz)
np.seterr(over='ignore')

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
    - get_hash: simülasyon sonundan deterministik, SHA-olmayan karıştırma ile byte dizisi üretir.
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

    # --------- Parametreleri seed'den üretme ---------
    def params_from_seed(self):
        u = self.u
        x0   = _map_u64(u[0], -7.5, -2.5)    # başlangıç merkezi
        k0   = _map_u64(u[1],  2.0,  8.0)    # momentum
        sigma = _map_u64(u[2], 0.4,  1.2)    # genişlik

        center = _map_u64(u[3], -0.5, 0.5)   # bariyer merkezi
        width  = _map_u64(u[4],  0.5,  2.5)  # bariyer genişliği
        V0     = _map_u64(u[5],  2.0, 15.0)  # bariyer yüksekliği

        absorb_strength = _map_u64(u[6], 0.15, 0.35)
        dt = _map_u64(u[7], 0.004, 0.012)

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
                    title="Tünelleme Simülasyonu", 
                    do_plot: bool = False):
        """
        Zaman evrimi (split-operator + absorbing boundary)
        do_plot=True verilirse grafik çizilir, False ise çizilmez
        """
        k = 2 * np.pi * np.fft.fftfreq(self.N, d=self.dx)
        T = np.exp(-1j * (self.hbar * k**2) / (2 * self.m) * dt / 2)
        Vop = np.exp(-1j * V * dt / self.hbar)
        mask = self.absorbing_boundary(absorb_strength)

        if do_plot:
            plt.figure()

        maxy = 0.0
        for n in range(Nt):
            psi_k = np.fft.fft(psi * mask)
            psi   = np.fft.ifft(T * psi_k)
            psi   = Vop * psi
            psi_k = np.fft.fft(psi)
            psi   = np.fft.ifft(T * psi_k)

            if do_plot and (n % 200 == 0 or n == Nt-1):
                y = np.abs(psi)**2
                maxy = max(maxy, y.max())
                plt.plot(self.x, y, label=f"t={n*dt:.2f}")

        if do_plot:
            Vscaled = (V / (V.max() if V.max() != 0 else 1.0)) * (0.9*maxy)
            plt.plot(self.x, Vscaled, "k--", label="V(x) scaled")
            plt.xlabel("x"); plt.ylabel(r"$|\psi(x)|^2$")
            plt.title(title); plt.legend(); plt.tight_layout(); plt.show()
            plt.close()

        self.psi = psi
        return psi

    # --------- Hash üretme (sınıf metodu) ---------
    def get_hash(self, output_bits: int = 256) -> bytes:
        """
        Yeni get_hash:
        - psi'den prob. yoğunluğu alır, kesinlikle normalize eder.
        - prob_density float64 bit-pattern'lerini uint64 olarak yorumlar.
        - Bu uint64 kelimeleri seed'den üretilmiş 8 uint64 ile etkileştirip
          SplitMix-esque bir karıştırıcı ile karıştırır.
        - Çıktıyı istenen bit sayısına kısaltır (deterministik, non-cryptographic mixing).
        Not: SHA veya dış kütüphane kullanılmaz.
        """
        if self.psi is None:
            raise ValueError("Önce simülasyonu çalıştırmalısınız (evolve_tdse).")

        # 1) Olasılık yoğunluğu ve kesin normalize
        prob_density = np.abs(self.psi)**2
        total_prob = np.sum(prob_density) * self.dx
        if total_prob == 0:
            raise ValueError("Toplam olasılık sıfır; simülasyonu kontrol et.")
        prob_density = prob_density / total_prob  # kesin normalize

        # 2) Float64 bit-pattern'lerini uint64 olarak al
        float_bytes = prob_density.astype(np.float64).tobytes()
        u64_from_prob = np.frombuffer(float_bytes, dtype=np.uint64).astype(np.uint64)

        # 3) Seed'e dayalı başlangıç durumu (self.u zaten 8 adet uint64)
        seed_u64 = np.array(self.u, dtype=np.uint64)  # 8 elemanlı

        # 4) SplitMix64 tarzı karıştırıcı (işlem uint64 sınırında)
        def splitmix64(x: np.uint64) -> np.uint64:
            mask64 = np.uint64(0xFFFFFFFFFFFFFFFF)
            x = (x + np.uint64(0x9E3779B97F4A7C15)) & mask64
            x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
            x = x & mask64
            x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
            x = x & mask64
            x = x ^ (x >> np.uint64(31))
            return x & mask64

        # 5) Karıştırma döngüsü: prob kelimelerini bloklar halinde indirger ve karıştırır
        out_bits = int(output_bits)
        if out_bits % 8 != 0:
            raise ValueError("output_bits 8'in katı olmalı.")
        out_bytes_len = out_bits // 8
        out_words_needed = (out_bytes_len + 7) // 8  # kaç 64-bit kelime gerekli

        words = []
        n = len(u64_from_prob)
        const = np.uint64(0x9E3779B97F4A7C15)
        mask64 = np.uint64(0xFFFFFFFFFFFFFFFF)

        if n == 0:
            # nadir ama garanti için: fallback seed türevi
            state = seed_u64.copy()
            for i in range(out_words_needed):
                v = splitmix64(np.uint64(state[i % 8] + np.uint64(i)))
                words.append(v)
        else:
            # chunk_size: prob kelimelerini kelime bazında böl
            chunk_size = max(1, n // out_words_needed)
            for wi in range(out_words_needed):
                start = wi * chunk_size
                end = min(start + chunk_size, n)
                block = u64_from_prob[start:end]
                # xor-reduce (çok hızlı) -> preserve entropy from all words in block
                xor_reduce = np.uint64(0)
                if block.size > 0:
                    xor_reduce = np.bitwise_xor.reduce(block)
                # mix with seed word and index to break symmetries
                mixed = xor_reduce ^ seed_u64[wi % seed_u64.size] ^ ((np.uint64(wi) * const) & mask64)
                # pass through splitmix for diffusion
                v = splitmix64(mixed)
                # ekstra nonlineer adım: rotate-left by lower 6 bits of v (basit)
                rot = int(v & np.uint64(0x3F))
                v = ((v << rot) | (v >> (64 - rot))) & mask64
                # son bir splitmix
                v = splitmix64(v)
                words.append(v)

        # 6) words -> byte dizisi ve uzunluğu istenene kısalt
        out_arr = np.array(words, dtype=np.uint64)
        out_bytes = out_arr.tobytes()[:out_bytes_len]

        return out_bytes


# --------- Yardımcılar: metinden seed ---------
def text_to_bits(text: str, encoding="utf-8") -> str:
    return ''.join(f"{byte:08b}" for byte in text.encode(encoding))

def seed_from_text(message: str) -> int:
    bit_str = text_to_bits(message)
    bit_str = bits_right_pad(bit_str, 512)
    return polynomial_calculate(bit_str)


# --------- Örnek kullanım / CLI ---------
if __name__ == "__main__":
    print("Creating seed and hash files (ASCII bitstreams)...")

    # Sonuçlar klasörünü oluştur
    os.makedirs("results", exist_ok=True)
    
    # 1000 adet seed üretmek için döngü
    num_seeds = 1000
    seeds_data = []
    #base_message = "Merhaba Quantum Hesh!"
    #base_seed_int = seed_from_text(base_message)
    
    for i in range(num_seeds):
        # Her döngüde farklı bir seed üretmek için base seed'e i değerini ekle
        seed_int =seed_from_text(str(i))
        seeds_data.append(seed_int)

    # --- 0–999 döngüsü ---
    with open("results/nist_seed_data.txt", "w") as sf, \
         open("results/nist_hash_data.txt", "w") as hf:
        
        for i in range(num_seeds):
            seed_int = seeds_data[i]
            
            th = TunnelingHash(seed_int=seed_int, N=2048, L=20.0)
            P = th.params_from_seed()
            psi0 = th.create_wave_packet(P["x0"], P["k0"], P["sigma"])
            V, _ = th.define_barrier_center_width(P["barrier_center"], P["barrier_width"], P["V0"])

            # Simülasyonu çalıştır (grafik yok)
            final_psi = th.evolve_tdse(
                psi0, V,
                dt=P["dt"], Nt=900,
                absorb_strength=P["absorb_strength"],
                do_plot=False
            )

            # Hash üret (512 bit)
            quantum_bytes = th.get_hash(output_bits=512)

            # seed ve hash'i 512-bit ikili string olarak kaydet
            seed_binary_string = format(seed_int, '0512b')
            quantum_binary_string = ''.join(format(byte, '08b') for byte in quantum_bytes)

            sf.write(seed_binary_string + "\n")
            hf.write(quantum_binary_string + "\n")

    print(f"Done. {num_seeds} adet farklı seed/hash ikilisi results/ klasörüne yazıldı.")
