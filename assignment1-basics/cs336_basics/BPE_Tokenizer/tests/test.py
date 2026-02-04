line = " Snap e"
last_space_idx = line.rfind(" ")
if last_space_idx == -1:
    raise ValueError(f"Invalid merge line (no space found): {line!r}")

token1_str = line[:last_space_idx]
token2_str = line[last_space_idx + 1:]

print((token1_str, token2_str))