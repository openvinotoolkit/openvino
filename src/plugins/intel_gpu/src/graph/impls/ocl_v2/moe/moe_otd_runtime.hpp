// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "intel_gpu/primitives/moe_3gemm_fused_compressed.hpp"
#include "intel_gpu/runtime/stream.hpp"
#include "lru_cache.hpp"
#include "openvino/util/parallel_io.hpp"

namespace ov::intel_gpu::ocl::moe_otd {

// Lightweight perf counters for OTD profiling.
// Enabled by setting MOE_OTD_PERF_LOG=1 environment variable.
// Counters are printed to stderr on process exit.
struct OtdPerfCounters {
    std::atomic<uint64_t> gpu_hits{0};
    std::atomic<uint64_t> gpu_misses{0};
    std::atomic<uint64_t> disk_io_ns{0};
    std::atomic<uint64_t> gpu_copy_ns{0};
    std::atomic<uint64_t> tensor_load_count{0};
    std::atomic<uint64_t> batched_fallbacks{0};
    std::atomic<uint64_t> grouped_fallbacks{0};
    std::atomic<uint64_t> created_onednn_kernels{0};

    void dump() const;
};

OtdPerfCounters* get_perf_counters();

class ParallelWeightReader {
public:
    explicit ParallelWeightReader(const std::filesystem::path& weights_path) : _weights_path(weights_path) {}

    ~ParallelWeightReader() {
        if (_opened) {
            ov::util::close_file_handle(_shared_handle);
        }
    }

    ParallelWeightReader(const ParallelWeightReader&) = delete;
    ParallelWeightReader& operator=(const ParallelWeightReader&) = delete;

    const std::filesystem::path& path() const {
        return _weights_path;
    }

    size_t file_size() {
        ensure_open();
        return _file_size;
    }

    void read(char* dst, size_t size, size_t file_offset) {
        ensure_open();
        if (!ov::util::positional_read(_shared_handle, dst, size, file_offset)) {
            OPENVINO_THROW("Failed to read enough bytes from OTD weight file");
        }
    }

private:
    void ensure_open() {
        if (!_opened) {
            std::streamoff file_size = 0;
            ov::util::get_file_handle_and_size(_weights_path, 0, _shared_handle, file_size);
            _file_size = static_cast<size_t>(file_size);
            _opened = true;
        }
    }

    std::filesystem::path _weights_path;
    ov::FileHandle _shared_handle{};
    size_t _file_size = 0;
    bool _opened = false;
};

// A constant sliced along its outer dimension (e.g. one slice per expert) whose slices are loaded on demand into a slot buffer
struct SlicedTensor {
    cldnn::memory::ptr slots;   // slice `i` of the buffer holds the slice loaded into slot `i`
    size_t slice_size = 0;      // bytes
    cldnn::memory::ptr source;  // device-resident tensor with every slice; null: the slices are read from the weights file
    size_t file_offset = 0;     // of slice 0 in the weights file, used without `source`
};

/// Loads slice `index` of each tensor into slot `slot` of its slot buffer.
/// Slices are read from the weights file as stored, or copied on the device from `source`. The file reads are complete on return;
/// the device copies are not waited for on an in-order queue, where the kernels that read the slots are enqueued after them.
void fill_weights_memory(cldnn::stream& exec_stream, ParallelWeightReader& weight_reader, const std::vector<SlicedTensor>& tensors, size_t index, size_t slot);

}  // namespace ov::intel_gpu::ocl::moe_otd