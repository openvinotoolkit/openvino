// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/openvino.hpp>

#include <cstdint>
#include <cstddef>
#include <cerrno>
#include <cstring>
#include <string>

#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

static bool write_all(int fd, const uint8_t* data, size_t size) {
    size_t off = 0;
    while (off < size) {
        ssize_t w = ::write(fd, data + off, size - off);
        if (w < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        off += static_cast<size_t>(w);
    }
    return true;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* Data, size_t Size) {
    // Keep target bounded & fast. libFuzzer will call this millions of times. [2](https://llvm.org/docs/LibFuzzer.html)[6](https://chromium.googlesource.com/chromium/src/testing/libfuzzer/+/f44bd1a3e684b72de7a90871a51ab4acb0df4054/efficient_fuzzer.md)
    constexpr size_t kMaxInput = 4 * 1024 * 1024; // 4 MiB
    if (Size == 0 || Size > kMaxInput) return 0;

    // Optional quick filter:
    // Many valid .tflite files begin with "TFL3" identifier (not guaranteed for all tooling),
    // but checking it can speed up early fuzzing. Consider enabling after you have a good seed corpus. [4](https://deepwiki.com/tensorflow/tflite-micro/4-model-format-and-optimization)
    // if (Size < 4 || std::memcmp(Data, "TFL3", 4) != 0) return 0;

    // Create temp directory (unique)
    char dir_tmpl[] = "/tmp/ov-tflite-XXXXXX";
    char* dir = ::mkdtemp(dir_tmpl);
    if (!dir) return 0;

    std::string model_path = std::string(dir) + "/model.tflite";

    int fd = ::open(model_path.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0600);
    if (fd < 0) {
        ::rmdir(dir);
        return 0;
    }

    (void)write_all(fd, Data, Size);
    ::close(fd);

    try {
        // Reuse Core for speed; recommended to avoid per-iteration heavy init. [6](https://chromium.googlesource.com/chromium/src/testing/libfuzzer/+/f44bd1a3e684b72de7a90871a51ab4acb0df4054/efficient_fuzzer.md)[2](https://llvm.org/docs/LibFuzzer.html)
        static ov::Core core;

        // API under test: reads IR/ONNX/PDPD/TF/TFLite; bin_path unused for .tflite. [1](https://docs.openvino.ai/2025/api/c_cpp_api/classov_1_1_core.html)
        auto model = core.read_model(model_path);

        // Touch some cheap accessors to exercise more of the post-load surface.
        if (model) {
            (void)model->get_name();
            (void)model->outputs();
        }
    } catch (const std::exception&) {
        // Invalid inputs are expected; swallow exceptions. libFuzzer target must tolerate any bytes. [2](https://llvm.org/docs/LibFuzzer.html)
    } catch (...) {
        // Never allow exceptions to escape the fuzz entrypoint. [2](https://llvm.org/docs/LibFuzzer.html)
    }

    ::unlink(model_path.c_str());
    ::rmdir(dir);
    return 0;
}
