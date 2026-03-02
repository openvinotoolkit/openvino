// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <openvino/openvino.hpp>   // ov::Core
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
    // Keep the target fast and bounded (libFuzzer runs in-process repeatedly).
    // You can tune this based on your infra.
    constexpr size_t kMaxInput = 4 * 1024 * 1024; // 4 MiB
    if (Size == 0 || Size > kMaxInput) return 0;

    // Create temp directory: /tmp/ov-onnx-XXXXXX
    char dir_tmpl[] = "/tmp/ov-onnx-XXXXXX";
    char* dir = ::mkdtemp(dir_tmpl);
    if (!dir) return 0;

    std::string model_path = std::string(dir) + "/model.onnx";

    int fd = ::open(model_path.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0600);
    if (fd < 0) {
        ::rmdir(dir);
        return 0;
    }

    (void)write_all(fd, Data, Size);
    ::close(fd);

    // Important: ov::Core::read_model reads IR/ONNX/etc from a model file path.
    // For ONNX, bin_path isn't used.
    try {
        static ov::Core core; // reuse across iterations (faster; avoids re-init)
        auto model = core.read_model(model_path); // fuzzed API under test
        // Optionally touch some metadata to exercise more code paths:
        if (model) {
            // Access model name / outputs etc. Keep it lightweight.
            (void)model->get_name();
            (void)model->outputs();
	    (void)model->inputs();
        }
    } catch (const std::exception&) {
        // Expected for malformed inputs; ignore.
    } catch (...) {
        // Never let exceptions escape the fuzz target.
    }

    // Cleanup
    ::unlink(model_path.c_str());
    ::rmdir(dir);
    return 0;
}