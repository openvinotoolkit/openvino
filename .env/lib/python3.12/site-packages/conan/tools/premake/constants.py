# Source: https://premake.github.io/docs/architecture/
CONAN_TO_PREMAKE_ARCH = {
    "x86": "x86",
    "x86_64": "x86_64",

    "armv4": "arm",
    "armv4i": "arm",
    "armv5el": "arm",
    "armv5hf": "arm",
    "armv6": "arm",
    "armv7": "arm",
    "armv7hf": "arm",
    "armv7s": "arm",
    "armv7k": "arm",

    "armv8": "arm64",
    "armv8_32": "arm64",
    "armv8.3": "arm64",
    "arm64ec": "arm64",

    "e2k-v2": "e2k",
    "e2k-v3": "e2k",
    "e2k-v4": "e2k",
    "e2k-v5": "e2k",
    "e2k-v6": "e2k",
    "e2k-v7": "e2k",

    "riscv64": "riscv64",

    "wasm": "wasm32",
    "wasm64": "wasm64",
    "asm.js": "wasm32",
}
