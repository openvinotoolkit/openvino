// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/graph/serialization/binary_buffer.hpp"
#include "intel_gpu/graph/serialization/cl_kernel_data_serializer.hpp"
#include "intel_gpu/graph/serialization/helpers.hpp"
#include "intel_gpu/graph/serialization/set_serializer.hpp"
#include "intel_gpu/graph/serialization/string_serializer.hpp"
#include "intel_gpu/graph/serialization/vector_serializer.hpp"
#include "intel_gpu/graph/program.hpp"

#include "primitive_inst.h"
#include "kernel_base.hpp"


#include <vector>
#include <utility>

namespace cldnn {
namespace ocl {

/*
Base class for all GPU implementation of specified primitive type.
For example, all gpu convolution implementations should derive from typed_primitive_impl_ocl_new<convolution>.
*/
template <class PType>
struct typed_primitive_impl_ocl_new : public typed_primitive_impl<PType> {
    std::vector<ov::intel_gpu::ocl::KernelData> _kernel_data;
    std::vector<kernel::ptr> _kernels;

    typed_primitive_impl_ocl_new() = default;

    typed_primitive_impl_ocl_new(const typed_primitive_impl_ocl_new<PType>& other)
    : typed_primitive_impl<PType>(other._weights_reorder_params, other._kernel_name, other._is_dynamic)
    , _kernel_data(other._kernel_data)
    , _kernels({}) {
        _kernels.reserve(other._kernels.size());
        for (size_t k = 0; k < other._kernels.size(); ++k) {
            _kernels.emplace_back(other._kernels[k]->clone(other.can_share_kernels));
        }
        this->m_manager = other.m_manager;
    }

    typed_primitive_impl_ocl_new(const std::vector<ov::intel_gpu::ocl::KernelData>& kd, const std::string& impl_name)
        : typed_primitive_impl<PType>(nullptr, impl_name),
          _kernel_data(kd) {
    }

    bool is_cpu() const override { return false; }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_impl::save(ob);
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_impl::load(ib);
    }

    void update(primitive_inst& inst, const kernel_impl_params& impl_params) override {
        auto new_impl_params = this->canonicalize_shapes(impl_params);
        update_dispatch_data(new_impl_params);
        inst.update_shape_info_tensor(new_impl_params);
    }

protected:
    virtual kernel_arguments_data get_arguments(const typed_primitive_inst<PType>& instance) const {
        kernel_arguments_data args;

        for (size_t i = 0; i < instance.inputs_memory_count(); i++) {
            args.inputs.push_back(instance.input_memory_ptr(i));
        }

        if (instance.has_fused_primitives()) {
            size_t count = instance.get_fused_mem_count();
            for (size_t i = 0; i < count; i++) {
                args.fused_op_inputs.push_back(instance.fused_memory(i));
            }
        }

        for (size_t i = 0; i < instance.outputs_memory_count(); i++) {
            args.outputs.push_back(instance.output_memory_ptr(i));
        }

        args.shape_info = instance.shape_info_memory_ptr();

        return args;
    }

    void init_kernels(const kernels_cache& kernels_cache, const kernel_impl_params& params) override {
        _kernels.clear();
        if (!_kernel_data.empty()) {
            auto compiled_kernels = kernels_cache.get_kernels(params);
            _kernels.insert(_kernels.begin(), compiled_kernels.begin(), compiled_kernels.end());
            // batch program hash and kernel entry point to find corresponding cl source code
            // kernel_dump_info = std::make_pair(std::to_string(kernels_cache.get_kernel_batch_hash(params)),
            //                               _kernel_data.kernels[0].code.kernelString->entry_point);
            // for (size_t i = 1; i < _kernel_data.kernels.size(); ++i)
            //     kernel_dump_info.second += " " + _kernel_data.kernels[i].code.kernelString->entry_point;
        }
    }

    void init_by_cached_kernels(const kernels_cache& kernels_cache, std::vector<std::string>& cached_kernel_ids) override {
        _kernels.clear();

        _kernels.reserve(cached_kernel_ids.size());
        for (size_t k = 0; k < cached_kernel_ids.size(); ++k) {
            _kernels.emplace_back(kernels_cache.get_kernel_from_cached_kernels(cached_kernel_ids[k]));
        }
        this->can_share_kernels = kernels_cache.get_kernels_reuse();
    }

    std::vector<std::string> get_cached_kernel_ids(const kernels_cache& kernels_cache) override {
        return {kernels_cache.get_cached_kernel_ids(_kernels)};
    }

    std::vector<kernel::ptr> get_kernels() const override {
        return _kernels;
    }

    std::vector<layout> get_internal_buffer_layouts_impl() const override {
        // if (_kernel_data.internalBufferSizes.empty())
            return {};

        // std::vector<layout> layouts;
        // auto dtype = from_data_type(_kernel_data.internalBufferDataType);
        // const auto bpp = data_type_traits::size_of(dtype);
        // for (auto size : _kernel_data.internalBufferSizes) {
        //     layout inbuf_layout = {dtype, format::bfyx, // simple linear format (flattern to x channel)
        //                             {1, 1, 1, (tensor::value_type)(size / bpp)}};
        //     layouts.push_back(inbuf_layout);
        // }
        // return layouts;
    }

    void set_arguments_impl(typed_primitive_inst<PType>& instance) override {
        // if (instance.can_be_optimized() || is_cpu()) {
        //     return;
        // }

        OPENVINO_ASSERT(_kernels.size() == _kernel_data.size(), "[GPU] Mismatch between compiled kernels count and expected kernels data\n",
                                                                "[GPU] Compiled kernels count: ", _kernels.size(), "\n",
                                                                "[GPU] KernelData count: ", _kernel_data.size(), "\n",
                                                                "[GPU] Likely some issue with empty tensor handling happened");

        stream& stream = instance.get_network().get_stream();
        for (size_t kd_idx = 0; kd_idx < _kernel_data.size(); ++kd_idx) {
            // if (_kernel_data.kernels[kd_idx].skip_execution) {
            //     continue;
            // }

            auto args = get_arguments(instance);
            args.scalars = &_kernel_data[kd_idx].params.scalars;

            for (const auto& m : instance.get_intermediates_memories()) {
                args.intermediates.push_back(m);
            }

            stream.set_arguments(*_kernels[kd_idx], _kernel_data[kd_idx].params, args);
        }
    }

    void set_arguments_impl(typed_primitive_inst<PType>& instance, kernel_arguments_data& args) override {
        // if (instance.can_be_optimized()) {
        //     return;
        // }

        stream& stream = instance.get_network().get_stream();

        for (size_t k = 0; k < _kernels.size(); ++k) {
            // if (_kernel_data[k].skip_execution)
            //     continue;

            stream.set_arguments(*_kernels[k], _kernel_data[k].params, args);
        }
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events,
                            typed_primitive_inst<PType>& instance) override {
        stream& stream = instance.get_network().get_stream();
        if (instance.can_be_optimized()) {
            return stream.aggregate_events(events, false, instance.is_output());
        }
        std::vector<event::ptr> tmp_events(events);
        std::vector<event::ptr> all_events;
        OPENVINO_ASSERT(_kernels.size() == _kernel_data.size(), "[GPU] Mismatch between compiled kernels count and expected kernels data\n",
                                                                "[GPU] Compiled kernels count: ", _kernels.size(), "\n",
                                                                "[GPU] KernelData count: ", _kernel_data.size(), "\n",
                                                                "[GPU] Likely some issue with empty tensor handling happened");
        for (size_t kd_idx = 0; kd_idx < _kernel_data.size(); ++kd_idx) {
            // if (_kernel_data.kernels[kd_idx].skip_execution)
            //     continue;
            // If any user of the prim's users is CPU implementation or network's output, set prim as a output event (event won't be nullptr)
            bool needs_completion_event = instance.needs_completion_event();

            auto& params = _kernel_data[kd_idx].params;
            auto args = get_arguments(instance);
            args.scalars = &params.scalars;

            for (const auto& m : instance.get_intermediates_memories()) {
                args.intermediates.push_back(m);
            }

            const auto& gws = params.workGroups.global;
            const auto& lws = params.workGroups.local;

            GPU_DEBUG_TRACE_DETAIL << "Enqueue kernel " << kd_idx << ": gws=[" << gws[0] << ", " << gws[1] << ", " << gws[2] << "] "
                                   << "lws=[" << lws[0] << ", " << lws[1] << ", " << lws[2] << "]"
                                   << (needs_completion_event ? " has_completion_event=true" : "") << std::endl;

            auto ev = stream.enqueue_kernel(*_kernels[kd_idx], params, args, tmp_events, needs_completion_event);
            // if (_kernel_data.needs_sub_kernels_sync) {
            //     tmp_events = {ev};
            // }
            all_events.push_back(ev);
        }

        if ((all_events.size() == 0) && (tmp_events.size() > 0))
            return stream.aggregate_events(tmp_events);

        bool group_events = (all_events.size() > 1);
        return stream.aggregate_events(all_events, group_events);
    }

    std::vector<std::shared_ptr<cldnn::kernel_string>> get_kernels_source() override {
        std::vector<std::shared_ptr<cldnn::kernel_string>> kernel_strings;
        for (size_t i = 0; i < _kernel_data.size(); ++i) {
            kernel_strings.push_back(_kernel_data[i].code.kernelString);
        }
        return kernel_strings;
    }

    void reset_kernels_source() override {
        // for (size_t i = 0; i < _kernel_data.kernels.size(); ++i) {
        //     _kernel_data.kernels[i].code.kernelString.reset();
        // }
    }

    void set_kernels(cldnn::kernels_cache::compiled_kernels kernels) override {
        OPENVINO_ASSERT(kernels.size() == 1, "Only the kernels of the single primitive should be allowed.");
        auto& kernel_vec = kernels.begin()->second;
        _kernels.clear();
        _kernels.resize(kernel_vec.size());
        for (auto& k : kernel_vec) {
            auto sub_kernel_idx = k.second;
            _kernels[sub_kernel_idx] = k.first;
        }
    }

    std::vector<kernel::ptr> get_kernels() override {
        return _kernels;
    }

    std::pair<std::string, std::string> get_kernels_dump_info() const override {
        return {};
        // return kernel_dump_info;
    }

    virtual void update_dispatch_data(const kernel_impl_params& impl_params) {
        OPENVINO_ASSERT(this->_is_dynamic, "[GPU] update_dispatch_data() is called for static shape implementation ", this-> _kernel_name);
        OPENVINO_ASSERT(false, "[GPU] update_dispatch_data() is not implemented for dynamic implemenation ", this->_kernel_name);
    }
};

}  // namespace ocl
}  // namespace cldnn
