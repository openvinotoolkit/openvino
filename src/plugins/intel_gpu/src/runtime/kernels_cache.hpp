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
#include "kernels_factory.hpp"
#include "ocl/ocl_engine.hpp"
#include "serialization/set_serializer.hpp"
#include "serialization/vector_serializer.hpp"
#include "serialization/map_serializer.hpp"
#include "serialization/string_serializer.hpp"

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

    template <typename BufferType>
    void save(BufferType& buffer) const {
        buffer << _prog_id;
        buffer << batch_header_str;
        buffer << serialize_info_container;
    }

    template <typename BufferType>
    void load(BufferType& buffer) {
        buffer >> serialize_info_container;
        _kernels.clear();
        std::unique_ptr<ocl::ocl_engine> build_engine = nullptr;
        if (_engine.type() == engine_types::ocl) {
            build_engine = make_unique<ocl::ocl_engine>(_engine.get_device(), runtime_types::ocl, _engine.configuration(), _engine.get_task_executor());
        }
        for (auto& si : serialize_info_container) {
            cl::Program::Binaries binary_kernels{si.precompiled_kernels};
            try {
                cl::vector<cl::Kernel> kernels;
                cl::Program program(build_engine->get_cl_context(), {build_engine->get_cl_device()}, binary_kernels);
                program.build(build_engine->get_cl_device(), si.build_options.c_str());
                program.createKernels(&kernels);
                {
                    std::lock_guard<std::mutex> lock(_mutex);
                    for (auto& k : kernels) {
                        const auto& entry_point = k.getInfo<CL_KERNEL_FUNCTION_NAME>();
                        const auto& k_id = si.entry_point_to_id.find(entry_point);
                        if (k_id != si.entry_point_to_id.end()) {
                            cl_kernel cl_kernel = k.get();
                            cl_context cl_context = build_engine->get_cl_context().get();
                            kernel::ptr kernel = kernels_factory::create(_engine, cl_context, cl_kernel, entry_point);
                            _kernels.insert({k_id->second, kernel});
                        } else {
                            throw std::runtime_error("Could not find entry point");
                        }
                    }
                }
            } catch (const cl::BuildError& err) {
                std::cout << "+++++ OpenCL build error" << std::endl;
            }
        }
    }

private:
    struct serialize_info {
        std::string build_options;
        std::map<std::string, std::string> entry_point_to_id;
        std::vector<unsigned char> precompiled_kernels;

        template <typename BufferType>
        void save(BufferType& buffer) const {
            buffer(build_options, entry_point_to_id, precompiled_kernels);
        }

        template <typename BufferType>
        void load(BufferType& buffer) {
            buffer(build_options, entry_point_to_id, precompiled_kernels);
        }
    };
    std::vector<serialize_info> serialize_info_container;
    const bool need_serialize = true;
    static std::mutex _mutex;
    engine& _engine;
    uint32_t _prog_id = 0;
    kernels_code _kernels_code;
    size_t _kernel_idx = 0;
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
    void remove_kernel(kernel_id id) {
        _kernels.erase(id);
    }
};

}  // namespace cldnn
