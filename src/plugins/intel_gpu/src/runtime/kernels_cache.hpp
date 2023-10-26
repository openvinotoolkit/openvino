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

#include "openvino/runtime/threading/cpu_streams_executor.hpp"
#include "kernels_factory.hpp"
#include "ocl/ocl_engine.hpp"

namespace cldnn {

class kernels_cache {
public:
    struct kernel_code {
        std::vector<std::shared_ptr<kernel_string>> kernel_strings;
        kernel_impl_params params;
        bool dump_custom_program;

        kernel_code(const std::vector<std::shared_ptr<kernel_string>>& _kernel_strings,
                    const kernel_impl_params& _params,
                    bool _dump_custom_program)
            : kernel_strings(_kernel_strings),
                params(_params),
                dump_custom_program(_dump_custom_program) {}
    };

    struct impl_hasher {
        size_t operator()(const kernel_impl_params &k) const {
            return k.hash();
        }
    };

    using kernels_code = std::unordered_map<kernel_impl_params, kernel_code, impl_hasher>;

    using source_code = std::vector<std::string>;
    struct batch_program {
        int32_t bucket_id;
        int32_t batch_id;
        size_t hash_value;
        uint32_t kernels_counter;
        source_code source;
        std::string options;
        bool dump_custom_program;
        std::map<std::string, std::pair<kernel_impl_params, size_t>> entry_point_to_id;

        explicit batch_program(int32_t _bucket_id, int32_t _batch_id, std::string _options, const std::vector<std::string>& batch_header_str)
            : bucket_id(_bucket_id),
              batch_id(_batch_id),
              hash_value(0),
              kernels_counter(0),
              source(std::move(batch_header_str)),
              options(_options),
              dump_custom_program(false),
              entry_point_to_id({}) {
        }
    };

    using compiled_kernels = std::unordered_map<kernel_impl_params, std::vector<std::pair<kernel::ptr, size_t>>, impl_hasher>;

private:
    static std::mutex _mutex;
    engine& _engine;
    std::shared_ptr<ov::threading::ITaskExecutor> _task_executor;
    ExecutionConfig _config;
    uint32_t _prog_id = 0;
    kernels_code _kernels_code;
    std::atomic<bool> _pending_compilation{false};
    compiled_kernels _kernels;
    std::map<std::vector<unsigned char>, uint32_t> _cached_binaries;
    std::unordered_map<std::string, kernel::ptr> _cached_kernels;
    std::vector<std::string> batch_header_str;
    std::unordered_map<kernel_impl_params, size_t, impl_hasher> _kernel_batch_hash;
    void get_program_source(const kernels_code& kernels_source_code, std::vector<batch_program>*) const;
    void build_batch(const engine& build_engine, const batch_program& batch, compiled_kernels& compiled_kernels);

    std::string get_cache_path() const;
    bool is_cache_enabled() const;
    size_t get_max_kernels_per_batch() const;

public:
    explicit kernels_cache(engine& engine,
                           const ExecutionConfig& config,
                           uint32_t prog_id,
                           std::shared_ptr<ov::threading::ITaskExecutor> task_executor = nullptr,
                           const std::vector<std::string>& batch_header_str = {});
    kernel::ptr get_kernel_from_cached_kernels(std::string id) const;
    std::vector<kernel::ptr> get_kernels(kernel_impl_params params) const;
    void set_batch_header_str(const std::vector<std::string> &batch_headers) {
        batch_header_str = std::move(batch_headers);
    }

    bool validate_simple_kernel_execution(kernel::ptr kernel);

    // forces compilation of all pending kernels/programs
    void build_all();
    void reset();

    void add_kernels_source(const kernel_impl_params& params,
                                const std::vector<std::shared_ptr<kernel_string>>& kernel_sources,
                                bool dump_custom_program = false);
    compiled_kernels compile(const kernel_impl_params& params,
                                const std::vector<std::shared_ptr<kernel_string>>& kernel_sources,
                                bool dump_custom_program = false);

    std::string get_cached_kernel_id(kernel::ptr kernel) const;
    std::vector<std::string> get_cached_kernel_ids(const std::vector<kernel::ptr>& kernels) const;
    void add_to_cached_kernels(const std::vector<kernel::ptr>& kernels);

    size_t get_kernel_batch_hash(const kernel_impl_params& params) const {
        if (_kernel_batch_hash.find(params) != _kernel_batch_hash.end())
            return _kernel_batch_hash.at(params);
        return 0;
    }

    void save(BinaryOutputBuffer& ob) const;
    void load(BinaryInputBuffer& ib);
};

}  // namespace cldnn
