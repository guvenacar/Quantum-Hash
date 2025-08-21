from model.tunneling_hash.tunneling_hash import TunnelingHash

if __name__ == "__main__":
    th = TunnelingHash(seed=42)
    print(th.hash("merhaba dünya"))
