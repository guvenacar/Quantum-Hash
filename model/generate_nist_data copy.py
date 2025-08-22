# dynamic_seed_generator.py
import os

def seed_from_bytes(bit_string: str) -> bytes:
    """
    ASCII bit string -> binary bytes
    """
    byte_array = bytearray()
    byte = 0
    bits_filled = 0
    for bit_char in bit_string:
        if bit_char not in ('0', '1'):
            continue
        byte = (byte << 1) | int(bit_char)
        bits_filled += 1
        if bits_filled == 8:
            byte_array.append(byte)
            byte = 0
            bits_filled = 0
    if bits_filled > 0:
        byte = byte << (8 - bits_filled)  # eksik baytı sola 0 ekle
        byte_array.append(byte)
    return bytes(byte_array)

def save_binary_from_ascii(ascii_file: str, binary_file: str):
    """
    ASCII formatındaki '0' ve '1'lerden oluşan bir dosyayı
    NIST STS uyumlu binary dosyasına çevirir.
    """
    with open(ascii_file, "r") as f_in, open(binary_file, "wb") as f_out:
        buffer = 0
        count = 0
        while True:
            char = f_in.read(1)
            if not char:
                break
            if char not in ("0", "1"):
                continue  # boşluk, satır sonu vs. atlanır
            buffer = (buffer << 1) | int(char)
            count += 1
            if count == 8:
                f_out.write(bytes([buffer]))
                buffer = 0
                count = 0
        # Kalan bitler (8 bitten az) varsa
        if count > 0:
            buffer = buffer << (8 - count)  # eksik bitleri 0 ile doldur
            f_out.write(bytes([buffer]))

if __name__ == "__main__":
    ascii_input = "results/nist_test_data.txt"
    binary_output = "results/nist_test_data.bin"
    save_binary_from_ascii(ascii_input, binary_output)
    print(f"Binary dosya oluşturuldu: {binary_output}")

