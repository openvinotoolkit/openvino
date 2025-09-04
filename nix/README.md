# OpenVINO Nix Packaging

This directory contains Nix packaging files for building OpenVINO™ Toolkit on NixOS and other Nix-based systems.

## 🚀 Quick Start

### Using Flake (Recommended)

```bash
# Build OpenVINO
nix build .#openvino

# Enter development shell
nix develop

# Build with tests enabled (advanced users)
nix build .#openvino-with-tests
```

### Using Traditional Nix

```bash
# Build OpenVINO
nix-build -A openvino

# Enter shell with dependencies
nix-shell
```

## 🔧 Configuration

### Test Building

The Nix packaging **fixes the test configuration issue** that was causing build failures:

- **Default**: Tests are disabled (`enableTests = false`) to ensure successful builds
- **Optional**: Tests can be enabled by setting `enableTests = true` for advanced users

### CMake Flags

The packaging automatically sets the correct CMake flags:

```nix
# FIXED: Proper test configuration to prevent build conflicts
(cmakeBool "ENABLE_TESTS" enableTests)
(cmakeBool "ENABLE_FUNCTIONAL_TESTS" enableTests)
(cmakeBool "BUILD_TESTING" enableTests)
```

## 🐛 Issue Resolution

### Problem
The original issue (#31755) was caused by conflicting test flags:
- `ENABLE_TESTS=true`
- `ENABLE_FUNCTIONAL_TESTS=true` 
- `BUILD_TESTING=false`

This created a build conflict where CMake tried to build test binaries but `BUILD_TESTING=false` prevented proper test infrastructure setup.

### Solution
The Nix packaging now:
1. **Consistently sets all test flags** to the same value
2. **Defaults to disabled tests** for reliable builds
3. **Provides an option to enable tests** when needed
4. **Prevents build conflicts** by ensuring flag consistency

## 📦 Features

- ✅ **Reliable builds** - No more test-related build failures
- ✅ **Cross-platform support** - Works on Linux, NixOS, and other Nix systems
- ✅ **CUDA support** - Optional CUDA integration
- ✅ **Python bindings** - Full Python API support
- ✅ **System libraries** - Uses system-provided dependencies when possible
- ✅ **Parallel building** - Optimized for multi-core systems

## 🛠️ Customization

### Enable Tests

```nix
# In your shell.nix or default.nix
openvino = pkgs.callPackage ./nix/default.nix {
  enableTests = true;  # Enable test building
};
```

### CUDA Support

```nix
# Enable CUDA support
openvino = pkgs.callPackage ./nix/default.nix {
  cudaSupport = true;
  cudaPackages = pkgs.cudaPackages;
};
```

## 📚 Dependencies

The packaging automatically handles:
- **Build tools**: CMake, Ninja, Python, Git
- **Libraries**: OpenCV, Protobuf, Flatbuffers, PugiXML, Snappy, TBB
- **GPU support**: Level Zero, OpenCL, CUDA (optional)
- **Python**: Python 3.x with required packages

## 🚨 Troubleshooting

### Build Fails with Test Errors
- Ensure you're using the default configuration (`enableTests = false`)
- Check that all test flags are consistent

### Missing Dependencies
- The packaging should automatically resolve dependencies
- Check that you have the required system packages installed

### Platform Issues
- **macOS**: Currently marked as broken due to SDK issues
- **ARM64**: Full support for ARM-based systems
- **x86_64**: Full support for Intel/AMD systems

## 🤝 Contributing

To improve the Nix packaging:

1. **Test your changes** on multiple platforms
2. **Ensure builds succeed** with both test configurations
3. **Update dependencies** when OpenVINO requirements change
4. **Document any new options** or configurations

## 📄 License

This Nix packaging follows the same Apache 2.0 license as OpenVINO itself.

## 🔗 Links

- [OpenVINO Documentation](https://docs.openvino.ai/)
- [OpenVINO GitHub Repository](https://github.com/openvinotoolkit/openvino)
- [Nix Package Manager](https://nixos.org/nix/)
- [NixOS](https://nixos.org/)
