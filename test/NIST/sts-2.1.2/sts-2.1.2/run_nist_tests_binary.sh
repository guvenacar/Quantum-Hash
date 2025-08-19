#test/NIST/sts-2.1.2/sts-2.1.2/run_nist_tests_binary.sh
#!/bin/bash

# Her bir bit akışının uzunluğu (Python kodundaki 32 bayt = 256 bit)
L=256

# Toplam bit akışı sayısı (Python kodundaki gibi 10.000)
N=1000

# Giriş dosyamızın yolu
INPUT_FILE="../../../../results/nist_test_data.bin"

# `assess` programının varlığını kontrol et
if [ ! -f "assess" ]; then
    echo "Hata: 'assess' çalıştırılabilir dosyası bulunamadı. Lütfen 'make' komutunu çalıştırarak derleyin."
    exit 1
fi

echo "NIST Testleri Başlatılıyor..."
echo "-----------------------------------"
echo "Toplam Akış Sayısı: ${N}"
echo "Her bir Akışın Uzunluğu: ${L} bit"
echo "Dosya Yolu: ${INPUT_FILE}"
echo "-----------------------------------"

# assess programını interaktif modda doğru parametrelerle çalıştırıyoruz.
(
    echo "0"  # Giriş dosyasını seç (Input File)
    echo "${INPUT_FILE}"
    echo "1"  # Tüm testleri uygula
    echo "0"  # Varsayılan parametrelerle devam et
    echo "${N}" # Bitstream sayısını gir
    echo "1"  # Formatı Binary olarak seç
) | ./assess "${L}"

echo "NIST testleri tamamlandı."

# Test tamamlandıktan sonra raporu kopyala
REPORT_DIR="./experiments/AlgorithmTesting"
if [ -f "${REPORT_DIR}/finalAnalysisReport.txt" ]; then
    cp "${REPORT_DIR}/finalAnalysisReport.txt" \
       "${REPORT_DIR}/finalAnalysisReport_binary.txt"
    echo "Binary test raporu kaydedildi: finalAnalysisReport_binary.txt"
else
    echo "Rapor dosyası bulunamadı: finalAnalysisReport.txt"
fi

