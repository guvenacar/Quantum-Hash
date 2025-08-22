#!/bin/bash

# Örnek olarak küçük bir bit akışı ile test
L=512        # Her akışın uzunluğu
N=100         # Toplam akış sayısı, büyük dosya yerine küçük
INPUT_FILE="../../../../results/test_quantum_hash.bin"

# assess programının varlığını kontrol et
if [ ! -f "assess" ]; then
    echo "Hata: 'assess' bulunamadı."
    exit 1
fi

echo "NIST testleri başlatılıyor (küçük test)..."
echo "-----------------------------------"
echo "Toplam Akış Sayısı: ${N}"
echo "Her bir Akışın Uzunluğu: ${L} bit"
echo "Dosya Yolu: ${INPUT_FILE}"
echo "-----------------------------------"

# assess'i interaktif modda pipe ile çalıştır
(
    echo "0"          # Input File seç
    echo "${INPUT_FILE}"
    echo "1"          # Tüm testleri uygula
    echo "0"          # Varsayılan parametreler
    echo "${N}"       # Bitstream sayısı
    echo "1"          # Format: Binary
) | ./assess "${L}"

echo "NIST testleri tamamlandı."
