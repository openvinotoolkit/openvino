// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#define ONEDNN_PRIMITIVE_SERIALIZATION

#include "primitive_inst.h"
#include "intel_gpu/graph/serialization/binary_buffer.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/file_util.hpp"
#include "to_string_utils.h"
#include "register.hpp"
#include "utils.hpp"
#include "runtime/ocl/ocl_event.hpp"

#include "quantize_inst.h"
#include "reorder_inst.h"

#include "reorder/reorder_weights_kernel_selector.h"
#include "reorder/reorder_kernel_base.h"
#include "impls/ocl/kernel_selector_helper.h"

#include <vector>
#include <list>
#include <utility>

#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_graph.hpp>
#include <oneapi/dnnl/dnnl_ocl.hpp>

using namespace dnnl::graph;

namespace cldnn {
namespace onednn {

static std::mutex cacheAccessMutex;

struct compiled_partition_info {
    dnnl::graph::partition partition;
    dnnl::graph::compiled_partition compiled_partition;
    std::vector<dnnl::graph::logical_tensor> inputs;
    std::vector<dnnl::graph::logical_tensor> outputs;
};

inline std::ostream& operator<<(std::ostream& os, const compiled_partition_info& info) {
    os << info.partition.get_id() << ",inputs[";

    auto in_it = info.inputs.begin();
    os << in_it->get_id() << ":" << in_it->get_dims();
    ++in_it;
    for ( ; in_it != info.inputs.end(); ++in_it) {
        os << "," << in_it->get_id() << ":" << in_it->get_dims();
    }

    os << "],outputs[";
    auto out_it = info.outputs.begin();
    os << out_it->get_id() << ":" << out_it->get_dims();
    ++out_it;
    for (; out_it != info.outputs.end(); ++out_it) {
        os << "," << out_it->get_id() << ":" << out_it->get_dims();
    }
    os << "]";
    return os;
}

/// Set any layout according to the connection relationship of partitions
///
/// @param partitions a list of partitions
/// @param id_to_set_any_layout a set of ids of logical tensors with any layout type
static void set_any_layout(const std::vector<dnnl::graph::partition> &partitions,
        std::unordered_set<size_t> &id_to_set_any_layout) {
    // mapping from output tensor id to the all supported flags of
    // supported partitions, we may only need outputs' supported flags
    std::unordered_map<size_t, std::vector<bool>> output_to_flag_map;
    for (const auto &p : partitions) {
        for (const auto &out : p.get_output_ports()) {
            size_t id = out.get_id();
            if (p.is_supported() &&
                output_to_flag_map.find(id) == output_to_flag_map.end()) {
                output_to_flag_map[id] = {};
            }
        }

        for (const auto &in : p.get_input_ports()) {
            size_t id = in.get_id();
            auto iter = output_to_flag_map.find(id);
            if (iter != output_to_flag_map.end()) {
                // collect all of supported flags of this tensor's uses
                // Considering we have such a graph:
                //
                //   partition_A  partition_B
                //        \           |
                //      tensor1    tensor2
                //           \     /     |
                //         partition_C  unsupported partition
                //              |
                //           tensor3
                //              |
                //          framework op
                //
                // so the mapping of partition_A's output will be { true }
                // the mapping of partition_B's output will be { true, false }
                // The mapping of partition_C's output will be { false }
                // Only when all supported flags are true, users can set any
                // layout.
                iter->second.push_back(p.is_supported());
            }
        }
    }

    for (const auto &p : partitions) {
        // no need to set `any` layout if this partition is not supported
        if (!p.is_supported()) continue;
        for (const auto &in : p.get_input_ports()) {
            size_t id = in.get_id();
            auto iter = output_to_flag_map.find(id);
            // if this input tensor is not an output of another supported
            // partition, just skip
            if (iter == output_to_flag_map.end()) continue;
            std::vector<bool> flag_vec = iter->second;
            // check if all of uses of this tensor are supported partitions,
            // if not, no need to set ANY layout.
            bool need_set_any = std::all_of(flag_vec.begin(), flag_vec.end(),
                    [](const bool a) { return a; });
            if (!need_set_any) continue;

            /// record the id of logical tensor that will be set to ANY layout
            id_to_set_any_layout.insert(id);
        }
    }
}

static logical_tensor::dims get_logical_tensor_dims(const cldnn::layout& layout) {
    auto cldnn_dims = layout.get_dims();
    logical_tensor::dims out_dims;
    std::transform(cldnn_dims.begin(), cldnn_dims.end(), std::back_inserter(out_dims),
        [](int32_t v) { return v; });
    return out_dims;
}

template <class PType>
struct typed_primitive_onednn_graph_impl : public typed_primitive_impl<PType> {
    const engine* _engine;
    bool _enable_profiling = false;
    std::vector<cldnn::layout> _input_layouts;
    cldnn::layout _output_layout;
    std::shared_ptr<dnnl::graph::graph> _graph;
    std::vector<compiled_partition_info> _compiled_partitions;
    std::unordered_map<size_t, logical_tensor> _concrete_tensors;
    std::unordered_map<size_t, logical_tensor> _id_to_queried_logical_tensors;
    std::unordered_map<uint32_t, std::unordered_map<size_t, dnnl::graph::tensor>> _args;
    std::vector<size_t> _intermediate_to_ids;

    typed_primitive_onednn_graph_impl(const engine& engine,
            const ExecutionConfig& config,
            const std::vector<cldnn::layout>& input_layouts,
            const cldnn::layout& output_layout)
        : typed_primitive_impl<PType>(), _engine(&engine),
            _input_layouts(input_layouts), _output_layout(output_layout) {
            _enable_profiling = config.get_property(ov::enable_profiling);
            GPU_DEBUG_GET_INSTANCE(debug_config);
            GPU_DEBUG_IF(!debug_config->dump_profiling_data.empty()) {
                _enable_profiling = true;
            }
    }

    typed_primitive_onednn_graph_impl(const engine& engine, const ExecutionConfig& config = {})
        : typed_primitive_impl<PType>({}, "undef"), _engine(&engine) {
            _enable_profiling = config.get_property(ov::enable_profiling);
            GPU_DEBUG_GET_INSTANCE(debug_config);
            GPU_DEBUG_IF(!debug_config->dump_profiling_data.empty()) {
                _enable_profiling = true;
            }
        }

    typed_primitive_onednn_graph_impl()
        : typed_primitive_impl<PType>({}, "undef"),
          _engine(nullptr) {
    }

    bool is_cpu() const override { return false; }
    bool is_onednn() const override { return true; }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_impl::save(ob);
#ifdef ONEDNN_PRIMITIVE_SERIALIZATION
        ob << _input_layouts;
        ob << _output_layout;
#endif
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_impl::load(ib);
#ifdef ONEDNN_PRIMITIVE_SERIALIZATION
        ib >> _input_layouts;
        ib >> _output_layout;
#endif
    }

    size_t get_last_partition_output_tensor_id() const {
        return (*_compiled_partitions.crbegin()->outputs.crbegin()).get_id();
    }

private:
    std::string get_cache_directory(const ExecutionConfig& config) const {
        auto path = config.get_property(ov::cache_dir);
        if (path.empty()) {
            return {};
        }

        if (path.back() != '/' && path.back() != '\\') {
            path += "/";
        }
        return path;
    }

    std::string generate_cache_path_from_key(const ExecutionConfig& config, std::vector<uint8_t> key) const {
        auto path = get_cache_directory(config);
        if (path.empty()) {
            return {};
        }

        std::string key_str(key.begin(), key.end());
        size_t hash = std::hash<std::string>()(key_str);
        return path + std::to_string(hash) + ".onednn.cl_cache";
    }

protected:
    virtual bool optimized_out(typed_primitive_inst<PType>&) const { return false; }

    virtual std::unordered_map<size_t, dnnl::graph::tensor> get_arguments(typed_primitive_inst<PType>& instance) const {
        std::unordered_map<size_t, dnnl::graph::tensor> args;
        auto& engine = instance.get_network().get_engine();
        auto dnnl_engine = engine.get_onednn_engine();

        /// concrete inputs
        for (size_t i = 0; i < instance.inputs_memory_count(); i++) {
            auto& input = instance.input_memory(i);
            auto& lt = _concrete_tensors.at(i);
            args[lt.get_id()] = dnnl::graph::tensor(lt, dnnl_engine, input.buffer_ptr());
        }

        /// intermediate buffers
        auto inter_memories = instance.get_intermediates_memories();
        // _intermediate_to_ids.size() == inter_memories.size()
        for (size_t i = 0; i < inter_memories.size(); i++) {
            auto lid = _intermediate_to_ids[i];
            auto& lt = _id_to_queried_logical_tensors.at(lid);
            args[lt.get_id()] = dnnl::graph::tensor(lt, dnnl_engine, inter_memories[i]->buffer_ptr());
        }

        /// output
        const auto last_output_id = get_last_partition_output_tensor_id();
        auto& output = instance.output_memory();
        auto& lt = _id_to_queried_logical_tensors.at(last_output_id);
        args[lt.get_id()] = dnnl::graph::tensor(lt, dnnl_engine, output.buffer_ptr());

        return args;
    }

    virtual std::unordered_map<size_t, dnnl::graph::tensor> get_arguments(typed_primitive_inst<PType>& instance, kernel_arguments_data& mem_args) const {
        std::unordered_map<size_t, dnnl::graph::tensor> args;
        return args;
    }

    void init_kernels(const kernels_cache&, const kernel_impl_params&) override { }

    void set_arguments_impl(typed_primitive_inst<PType>& instance) override {
        if (instance.can_be_optimized())
            return;
        uint32_t net_id = instance.get_network().get_id();
        _args[net_id] = get_arguments(instance);
    }

    void set_arguments_impl(typed_primitive_inst<PType>& instance, kernel_arguments_data& args) override {
        if (instance.can_be_optimized()) {
            return;
        }
        _args[instance.get_network().get_id()] = get_arguments(instance, args);
    }

    event::ptr execute_impl(const std::vector<event::ptr>& /* events */,
                            typed_primitive_inst<PType>& instance) override {
        auto& network = instance.get_network();
        auto& stream = network.get_stream();
        auto net_id = network.get_id();
        event::ptr event;

        if (!instance.can_be_optimized()) {
            for (auto& cpi : _compiled_partitions) {
                std::vector<dnnl::graph::tensor> inputs_ts, outputs_ts;
                std::transform(cpi.inputs.begin(), cpi.inputs.end(), std::back_inserter(inputs_ts),
                    [&net_id, this](dnnl::graph::logical_tensor const& lt) {
                        return this->_args[net_id][lt.get_id()];
                    });
                std::transform(cpi.outputs.begin(), cpi.outputs.end(), std::back_inserter(outputs_ts),
                    [&net_id, this](dnnl::graph::logical_tensor const& lt) {
                        return this->_args[net_id][lt.get_id()];
                    });
                cpi.compiled_partition.execute(stream.get_onednn_stream(), inputs_ts, outputs_ts);
            }
        }

        if (instance.needs_completion_event())
            event = stream.enqueue_marker({});

        // if (_enable_profiling) {
        //     if (instance.can_be_optimized()) {
        //         event = stream.create_user_event(true);
        //     } else {
        //         dnnl::reset_profiling(stream.get_onednn_stream());
        //     }
        // }

        // if (!instance.can_be_optimized()) {
        //     try {
        //         _prim.execute(stream.get_onednn_stream(), _args[net_id]);
        //     } catch (dnnl::error& err) {
        //         /// WA: Force exit. Any opencl api call can be hang after CL_OUT_OF_RESOURCES.
        //         if (err.status == dnnl_status_t::dnnl_out_of_memory) {
        //             ov::intel_gpu::ForceExit();
        //         }
        //         throw;    // rethrowing dnnl::error if not out_of_memory
        //     }

        //     if (_enable_profiling) {
        //         // Call wait() function here instead of finish() to prevent cache flushing,
        //         // this synchronization point is needed for correct OneDNN's profiling process
        //         stream.wait();

        //         std::vector<uint64_t> duration = dnnl::get_profiling_data(stream.get_onednn_stream(), dnnl::profiling_data_kind::time);
        //         OPENVINO_ASSERT(duration.size() == 1, "[GPU] oneDNN profiling data is expected to have info only for single primitive ",
        //                                               "actual number is ", duration.size());

        //         event = std::make_shared<ocl::ocl_event>(duration[0]);
        //     } else {
        //         // If oneDNN primitive is the output primitive or it's user is CPU implementation, then enqueue marker
        //         // with empty events wait list (which will trigger wait for all previously enqueued tasks) and
        //         // return it as oneDNN primitive's event as it is a single option for proper synchronization
        //         if (instance.needs_completion_event())
        //             event = stream.enqueue_marker({});
        //     }
        // }

        return event;
    }

    std::vector<layout> get_internal_buffer_layouts_impl() const override {
        std::vector<layout> internal_buffer_layouts;
        for (const auto& id : _intermediate_to_ids) {
            auto& lt = _id_to_queried_logical_tensors.at(id);
            internal_buffer_layouts.push_back({{1, 1, 1, (tensor::value_type)(lt.get_mem_size())},
                cldnn::data_types::u8, format::bfyx});
        }
        return internal_buffer_layouts;
    }
};

}  // namespace onednn
}  // namespace cldnn
