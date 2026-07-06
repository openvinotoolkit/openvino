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
    std::atomic<uint64_t> transpose_ns{0};
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
    explicit ParallelWeightReader(const std::filesystem::path& weights_path) : _weights_path(weights_path) {
        std::streamoff file_size = 0;
        ov::util::get_file_handle_and_size(weights_path, 0, _shared_handle, file_size);
        _file_size = static_cast<size_t>(file_size);
    }

    ~ParallelWeightReader() {
        ov::util::close_file_handle(_shared_handle);
    }

    const std::filesystem::path& path() const {
        return _weights_path;
    }

    size_t file_size() const {
        return _file_size;
    }

    void read(char* dst, size_t size, size_t file_offset) {
        if (!ov::util::positional_read(_shared_handle, dst, size, file_offset)) {
            throw std::runtime_error("Failed to read enough bytes from OTD weight file");
        }
    }

private:
    std::filesystem::path _weights_path;
    ov::FileHandle _shared_handle{};
    size_t _file_size = 0;
};

ParallelWeightReader& get_thread_local_weight_reader(const std::filesystem::path& weights_path);

void maybe_transpose_scale_zp(const cldnn::MOECompressed::Config& config,
                              const char* tensor_name,
                              const cldnn::layout& layout,
                              std::vector<uint8_t>& payload,
                              size_t per_expert_size);

void fill_weights_memory(cldnn::stream& exec_stream,
                         const cldnn::MOECompressed::Config& config,
                         const std::vector<size_t>& weight_bin_offsets,
                         const std::filesystem::path& weights_path,
                         cldnn::moe_weights& wei_mem,
                         const std::vector<uint32_t>& experts_list,
                         const std::vector<size_t>& lru_experts);

}  // namespace ov::intel_gpu::ocl::moe_otd