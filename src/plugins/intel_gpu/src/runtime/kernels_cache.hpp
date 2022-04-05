// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/kernel.hpp"

#include <map>
#include <mutex>
#include <vector>
#include <memory>
#include <atomic>
#include <string>
#include <set>

#include <threading/ie_cpu_streams_executor.hpp>

namespace cldnn {
class kernels_cache {
public:
    using source_code = std::vector<std::string>;
    struct batch_program {
        int32_t bucket_id;
        int32_t batch_id;
        size_t hash_value;
        uint32_t kernels_counter;
        source_code source;
        std::string options;
        bool dump_custom_program;
        std::map<std::string, std::string> entry_point_to_id;

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

    struct kernel_code {
        std::shared_ptr<kernel_string> kernel_strings;
        std::string id;
        bool dump_custom_program;
        size_t hash_value;

        kernel_code(const std::shared_ptr<kernel_string>& _kernel_strings,
                    const std::string& _id,
                    bool _dump_custom_program)
            : kernel_strings(_kernel_strings),
              id(_id),
              dump_custom_program(_dump_custom_program),
              hash_value(_kernel_strings->get_hash()) {}

        bool operator == (const kernel_code& rhs) const {
            return (hash_value == rhs.hash_value);
        }
    };

    struct cmp_kernel_code {
        bool operator()(const kernel_code& x1, const kernel_code& x2) const {
            return (x1.hash_value < x2.hash_value);
        }
    };

    using kernels_code = std::set<kernel_code, cmp_kernel_code>;

private:
    static std::mutex _mutex;
    engine& _engine;
    uint32_t _prog_id = 0;
    kernels_code _kernels_code;
    std::atomic<bool> _pending_compilation{false};
    std::map<const std::string, kernel::ptr> _kernels;
    std::vector<std::string> batch_header_str;

    void get_program_source(const kernels_code& kernels_source_code, std::vector<batch_program>*) const;
    void build_batch(const engine& build_engine, const batch_program& batch);

    std::string get_cache_path() const;
    bool is_cache_enabled() const;
    size_t get_max_kernels_per_batch() const;

public:
    explicit kernels_cache(engine& engine, uint32_t prog_id, const std::vector<std::string>& batch_header_str = {});
    kernel_id set_kernel_source(const std::shared_ptr<kernel_string>& kernel_string,
                                bool dump_custom_program);
    kernel::ptr get_kernel(kernel_id id) const;
    void set_batch_header_str(const std::vector<std::string> &batch_headers) {
        batch_header_str = std::move(batch_headers);
    }
    // forces compilation of all pending kernels/programs
    void build_all();
    void reset();
};

}  // namespace cldnn
