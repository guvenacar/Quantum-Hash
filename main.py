# main.py - Quantum-Hash projesi için varsayılan ana dosya
# Bu dosya, projenin Python tabanlı olduğunu gösterir.

import sys
import argparse
from pathlib import Path

# Projenizin çekirdek hash algoritması modüllerini buraya ekleyin
# Örneğin: from src.qthash import QuantumHash

def main():
    """
    Programın ana giriş noktası.
    """
    print("------------------------------------------")
    print("  Quantum-Hash: Kuantum Sıçrama Fonksiyonu")
    print("------------------------------------------")
    print("\nProje hakkında daha fazla bilgi için:")
    print("https://github.com/guvenacar/Quantum-Hash")
    print("\nBu dosya, projenizin ana dilini belirtmek için bir yer tutucudur.")
    print("Gelecekte, bu dosyayı komut satırı arayüzü (CLI) veya")
    print("bir demo uygulaması için kullanabilirsiniz.")

if __name__ == "__main__":
    main()
