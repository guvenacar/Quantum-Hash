#test/NIST/sts-2.1.2/sts-2.1.2/run_nist_tests_ascii.sh
#!/bin/bash

# Bit akışının uzunluğunu (51200 bit) parametre olarak belirtiyoruz.
STREAM_LENGTH=51200

# Giriş dosyamızın yolunu belirtiyoruz.
INPUT_FILE="../../../../results/nist_hash_data.txt"

# Testi çalıştır
(
echo "0"
echo "${INPUT_FILE}"
echo "1"
echo "0"
echo "1"
echo "0"
) | ./assess "${STREAM_LENGTH}"

# Test tamamlandıktan sonra raporu kopyala
REPORT_DIR="./experiments/AlgorithmTesting"
if [ -f "${REPORT_DIR}/finalAnalysisReport.txt" ]; then
    cp "${REPORT_DIR}/finalAnalysisReport.txt" \
        "${REPORT_DIR}/finalAnalysisReport_hash_ascii_test.txt"
    echo "ASCII test raporu kaydedildi: finalAnalysisReport_hash_ascii_test.txt"
else
    echo "Rapor dosyası bulunamadı: finalAnalysisReport.txt"
fi
