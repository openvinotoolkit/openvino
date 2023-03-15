// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/graph/serialization/binary_buffer.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/kernel.hpp"
#include "intel_gpu/runtime/execution_config.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"

#include <map>
#include <mutex>
#include <vector>
#include <memory>
#include <atomic>
#include <string>
#include <set>

#include <threading/ie_cpu_streams_executor.hpp>
#include "kernels_factory.hpp"
#include "ocl/ocl_engine.hpp"

namespace cldnn {

class kernels_cache {
public:
    struct kernel_key {
        kernel_impl_params params;
        size_t sub_kernel_idx;

        kernel_key(kernel_impl_params& _params,
                        size_t _sub_kernel_idx)
                : params(_params),
                    sub_kernel_idx(_sub_kernel_idx) {}

        size_t hash() const {
            return cldnn::hash_combine(params.hash(), sub_kernel_idx);
        }

        bool operator==(const kernel_key& rhs) const {
            if ((params == rhs.params)
                    && (sub_kernel_idx == rhs.sub_kernel_idx))
                return true;
            return false;
        }
    };

    struct kernel_code {
        std::shared_ptr<kernel_string> kernel_strings;
        kernel_key key;
        bool dump_custom_program;

        kernel_code(const std::shared_ptr<kernel_string>& _kernel_strings,
                    const kernel_key& _key,
                    bool _dump_custom_program)
            : kernel_strings(_kernel_strings),
                key(_key),
                dump_custom_program(_dump_custom_program) {}
    };

    struct impl_hasher {
        size_t operator()(const kernel_key &k) const {
            return k.hash();
        }
    };

    using kernels_code = std::unordered_map<kernel_key, kernel_code, impl_hasher>;

    using source_code = std::vector<std::string>;
    struct batch_program {
        int32_t bucket_id;
        int32_t batch_id;
        size_t hash_value;
        uint32_t kernels_counter;
        source_code source;
        std::string options;
        bool dump_custom_program;
        std::map<std::string, kernel_key> entry_point_to_key;

        explicit batch_program(int32_t _bucket_id, int32_t _batch_id, std::string _options, const std::vector<std::string>& batch_header_str)
            : bucket_id(_bucket_id),
              batch_id(_batch_id),
              hash_value(0),
              kernels_counter(0),
              source(std::move(batch_header_str)),
              options(_options),
              dump_custom_program(false),
              entry_point_to_key({}) {
        }
    };

    using compiled_kernels = std::unordered_map<kernel_key, kernel::ptr, impl_hasher>;

private:
    static std::mutex _mutex;
    engine& _engine;
    InferenceEngine::CPUStreamsExecutor::Ptr _task_executor;
    ExecutionConfig _config;
    uint32_t _prog_id = 0;
    kernels_code _kernels_code;
    std::atomic<bool> _pending_compilation{false};
    compiled_kernels _kernels;
    std::vector<std::string> batch_header_str;

    void get_program_source(const kernels_code& kernels_source_code, std::vector<batch_program>*) const;
    void build_batch(const engine& build_engine, const batch_program& batch, compiled_kernels& compiled_kernels);

    std::string get_cache_path() const;
    bool is_cache_enabled() const;
    size_t get_max_kernels_per_batch() const;

public:
    explicit kernels_cache(engine& engine,
                           const ExecutionConfig& config,
                           uint32_t prog_id,
                           InferenceEngine::CPUStreamsExecutor::Ptr task_executor = nullptr,
                           const std::vector<std::string>& batch_header_str = {});
    kernel::ptr get_kernel(kernel_impl_params params, size_t sub_kernel_idx = 0) const;
    void set_batch_header_str(const std::vector<std::string> &batch_headers) {
        batch_header_str = std::move(batch_headers);
    }

    bool validate_simple_kernel_execution(kernel::ptr kernel);

    // forces compilation of all pending kernels/programs
    void build_all();
    void reset();

    void add_kernels_source(kernel_impl_params& params,
                                std::vector<std::shared_ptr<kernel_string>> kernel_sources,
                                bool dump_custom_program = false);
    compiled_kernels compile(kernel_impl_params params,
                                std::vector<std::shared_ptr<kernel_string>> kernel_sources,
                                bool dump_custom_program = false);

    void add_kernels(const std::vector<std::string>& kernel_ids, const std::vector<kernel::ptr>& kernels);
    void save(BinaryOutputBuffer& ob) const;
    void load(BinaryInputBuffer& ib);
};

}  // namespace cldnn
