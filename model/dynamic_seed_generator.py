import os

def polynomial_calculate(bit_string: str, bit_length: int = 512) -> int:
    """
    Calculate a dynamic polynomial value from a bit string.
    The result is constrained to 512 bits.
    """
    # Pad or trim the input to ensure it matches the expected bit length
    if len(bit_string) < bit_length:
        bit_string = bit_string.ljust(bit_length, '0')
    elif len(bit_string) > bit_length:
        bit_string = bit_string[:bit_length]

    # Determine two base values for polynomial calculation
    base1 = ((2 * bit_string.count('1') + 3) % 511) | 3
    base0 = ((2 * (bit_length - bit_string.count('1')) + 3) % 511) | 3

    def calculate(bits: str, base: int) -> int:
        """
        Internal helper to compute a polynomial-like value
        from the bit string using the given base.
        """
        total = 0
        for bit in bits:
            coefficient = (base - 1) if bit == '1' else (base - 2)
            total = total * base + coefficient
        return total

    total0 = calculate(bit_string, base0)
    total1 = calculate(bit_string, base1)

    # Combine results and constrain to 512-bit space
    return (total0 * total1) % (2**512)


def bits_right_pad(bit_string: str, total_bits: int = 512) -> str:
    """
    Pad a binary string on the right with zeros to reach the desired length.
    Raises ValueError if the string contains characters other than '0' or '1'.
    """
    if not all(c in '01' for c in bit_string):
        raise ValueError("Invalid bit string – only '0' and '1' are allowed")
    return bit_string.ljust(total_bits, '0')[:total_bits]


if __name__ == "__main__":
    output_dir = "results"
    output_file = os.path.join(output_dir, "nist_test_data.bin")
    os.makedirs(output_dir, exist_ok=True)

    print("Creating binary file, please wait...")
    with open(output_file, "wb") as f:
        for i in range(1_000):
            try:
                # Convert integer to 512-bit binary string
                bit_str = bin(i)[2:].zfill(512)
                int_value = polynomial_calculate(bit_str)

                print("Input: ", i)
                print("Bit value: ", bin(int_value)[2:])
                print("Integer value: ", int_value)

                # Convert integer to bytes (64 bytes = 512 bits)
                binary_data = int_value.to_bytes(64, byteorder='big')
                f.write(binary_data)

            except Exception as e:
                print(f"Error processing input {i}: {e}")
                break

    print(f"Process complete. Total file size: {os.path.getsize(output_file)} bytes")
