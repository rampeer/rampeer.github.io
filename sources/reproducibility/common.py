import os.path
import struct


def assert_same_across_runs(metric, value):
    filename = metric + ".bin"
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            old_value, = struct.unpack("f", f.read())
        if abs(old_value - value) > 1e-8:
            print(f"{metric} is inconsistent! {old_value} != {value}")
        else:
            print(f"{metric} is consistent across runs: {value}")
    else:
        print(
            f"Cannot ensure consistency of {metric} between runs because it is the first run. Please run this script again.")

    with open(filename, "wb+") as f:
        f.write(struct.pack("f", value))
