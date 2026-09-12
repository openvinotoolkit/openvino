# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import struct
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an Intel GPU Vulkan SPIR-V C++ header")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("array_name")
    parser.add_argument("source_name")
    args = parser.parse_args()

    binary = args.input.read_bytes()
    if not binary or len(binary) % 4 != 0:
        parser.error("input must be a non-empty SPIR-V binary aligned to 32-bit words")

    words = struct.unpack(f"<{len(binary) // 4}I", binary)
    lines = []
    for offset in range(0, len(words), 8):
        lines.append("    " + ", ".join(f"0x{word:08x}U" for word in words[offset : offset + 8]) + ",")

    output = f"""// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

namespace cldnn {{
namespace vulkan {{

// Generated from {args.source_name}; validated for Vulkan 1.3.
inline constexpr uint32_t {args.array_name}[] = {{
{chr(10).join(lines)}
}};

}}  // namespace vulkan
}}  // namespace cldnn
"""
    args.output.write_text(output)


if __name__ == "__main__":
    main()
