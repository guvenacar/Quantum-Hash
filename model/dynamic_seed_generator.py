import os

def polinom_hesapla(bit_dizisi: str, bit_sayisi: int = 512) -> int:
    if len(bit_dizisi) < bit_sayisi:
        bit_dizisi = bit_dizisi.ljust(bit_sayisi, '0')
    elif len(bit_dizisi) > bit_sayisi:
        bit_dizisi = bit_dizisi[:bit_sayisi]

    
    taban1 = ((2 * bit_dizisi.count('1') + 3) % 511) | 3
    taban0 = ((2 * (bit_sayisi - bit_dizisi.count('1')) + 3) % 511) | 3

    def hesapla(bit_str, taban):
        toplam = 0
        for bit in bit_str:
            katsayi = (taban - 1) if bit == '1' else (taban - 2)
            toplam = toplam * taban + katsayi
        return toplam

    toplam0 = hesapla(bit_dizisi, taban0)
    toplam1 = hesapla(bit_dizisi, taban1)

    return (toplam0 * toplam1) % (2**512)


def bits_right_pad(bit_str: str, total_bits=512) -> str:
    if not all(c in '01' for c in bit_str):
        raise ValueError("Geçersiz bit dizisi - sadece '0' ve '1' içermeli")
    return bit_str.ljust(total_bits, '0')[:total_bits]


if __name__ == "__main__":
    output_dir = "sonuclar"
    output_file = os.path.join(output_dir, "nist_test_data.bin")
    os.makedirs(output_dir, exist_ok=True)

    print("İkili dosya oluşturuluyor, lütfen bekleyiniz...")
    with open(output_file, "wb") as f:
        for i in range(1_000):
            try: 
                bit_str = bin(i)[2:].zfill(512)  # 512-bit uzunluğunda
                int_value = polinom_hesapla(bit_str)
                print("Girdi: ", i)
                print("Bit değeri: ", bin(int_value)[2:] )
                print("Tam sayı değeri: ", int_value)
                # Tam sayıyı byte dizisine çevir (64 byte = 512 bit)
                binary_data = int_value.to_bytes(64, byteorder='big')

                f.write(binary_data)
                
            except Exception as e:
                print(f"Hata! i={i} değeri işlenirken bir sorun oluştu: {e}")
                break
            
    print(f"İşlem tamamlandı. Toplam dosya boyutu: {os.path.getsize(output_file)} byte")

 