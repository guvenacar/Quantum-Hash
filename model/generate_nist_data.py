import os
from quantum_tunnelling.quantum_tunneling_hash import TunnelingHash
from dynamic_seed_generator import polynomial_calculate, bits_right_pad


output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

seed_file = os.path.join(output_dir, "nist_seed_data.txt")
hash_file = os.path.join(output_dir, "nist_hash_data.txt")

print("Creating seed and hash files (ASCII, 512-bit each)...")

with open(seed_file, "w") as sf, open(hash_file, "w") as hf:
    for i in range(1000):
        # 512-bit input
        bit_str = bin(i)[2:]
        bit_str = bits_right_pad(bit_str, 512)

        # polynomial_calculate ile seed int üret
        int_value = polynomial_calculate(bit_str)
        seed_bits = bin(int_value)[2:].zfill(512)

        # TunnelingHash üzerinden quantum hash hesapla
        th = TunnelingHash(seed_int=int_value, N=2048, L=20.0)
        quantum_bytes = th.get_hash(output_bits=512)  # 512-bit hash
        hash_bits = bin(int.from_bytes(quantum_bytes, "big"))[2:].zfill(512)

        # dosyalara yaz
        sf.write(seed_bits + "\n")
        hf.write(hash_bits + "\n")

print(f"Seed file size: {os.path.getsize(seed_file)} bytes")
print(f"Hash file size: {os.path.getsize(hash_file)} bytes")
