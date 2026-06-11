// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/openvino.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

// Keep the same delimiter style as other OpenVINO fuzzers (e.g. import_model-fuzzer.cc). [1](https://github.com/openvinotoolkit/openvino/blob/master/tests/fuzz/src/import_model-fuzzer.cc)
static const char kSplitSequence[] = {
    'F','U','Z','Z','_','N','E','X','T','_','F','I','E','L','D'
};

static bool find_split(const uint8_t* data, size_t size, size_t& split_pos) {
    const size_t sep_size = sizeof(kSplitSequence);
    if (size < sep_size) return false;

    for (size_t i = 0; i + sep_size <= size; ++i) {
        if (std::memcmp(data + i, kSplitSequence, sep_size) == 0) {
            split_pos = i;
            return true;
        }
    }
    return false;
}

static bool write_file(const fs::path& p, const uint8_t* buf, size_t len) {
    std::ofstream out(p, std::ios::binary | std::ios::trunc);
    if (!out) return false;
    if (len) out.write(reinterpret_cast<const char*>(buf), static_cast<std::streamsize>(len));
    return out.good();
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    // We expect:  <xml-bytes> + kSplitSequence + <bin-bytes>
    const size_t sep_size = sizeof(kSplitSequence);
    if (size < sep_size + 1) return 0;

    size_t split_pos = 0;
    if (!find_split(data, size, split_pos)) return 0;

    const uint8_t* xml_buf = data;
    const size_t   xml_len = split_pos;

    const uint8_t* bin_buf = data + split_pos + sep_size;
    const size_t   bin_len = (split_pos + sep_size <= size) ? (size - (split_pos + sep_size)) : 0;

    // If you want to allow "XML only" cases, remove the bin_len check.
    // But the request explicitly wants xml+bin separated by the delimiter.
    if (xml_len == 0 || bin_len == 0) return 0;

    // Prepare a stable temp directory for this fuzzer.
    // (Using one directory prevents spraying random files all over /tmp.)
    fs::path dir;
    try {
        dir = fs::temp_directory_path() / "openvino_read_model_ir_fuzz";
        std::error_code ec;
        fs::create_directories(dir, ec);
        if (ec) return 0;
    } catch (...) {
        return 0;
    }

    static std::atomic<uint64_t> g_id{0};
    const uint64_t id = ++g_id;

    const std::string base = "model_" + std::to_string(id);
    const fs::path xml_path = dir / (base + ".xml");
    const fs::path bin_path = dir / (base + ".bin");

    if (!write_file(xml_path, xml_buf, xml_len)) return 0;
    if (!write_file(bin_path, bin_buf, bin_len)) {
        std::error_code ec;
        fs::remove(xml_path, ec);
        return 0;
    }

    try {
        ov::Core core;

        // IMPORTANT: Call read_model with *only* the XML path.
        // Per API docs: if bin_path is empty, it will try same-name .bin next to the .xml. [2](https://docs.openvino.ai/2025/api/c_cpp_api/classov_1_1_core.html)
        (void)core.read_model(xml_path.string());
    } catch (const std::exception&) {
        // Expected for malformed IR; fail gracefully.
    } catch (...) {
        // Also fail gracefully on any non-std exceptions.
    }

    // Cleanup to avoid disk blow-up during long fuzz runs.
    {
        std::error_code ec;
        fs::remove(xml_path, ec);
        fs::remove(bin_path, ec);
    }

    return 0;
}