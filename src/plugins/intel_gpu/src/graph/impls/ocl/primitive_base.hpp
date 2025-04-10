// Copyright (C) 2018-2025 Intel Corporation
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
#include "kernel_selector_helper.h"
#include "register.hpp"
#include "registry/implementation_map.hpp"
#include "concatenation_inst.h"
#include "gather_inst.h"
#include "permute_inst.h"
#include "strided_slice_inst.h"
#include "broadcast_inst.h"
#include "scatter_update_inst.h"
#include "scatter_elements_update_inst.h"
#include "scatter_nd_update_inst.h"

#include <vector>
#include <list>
#include <utility>

namespace cldnn {
namespace ocl {

/*
Base class for all GPU implementation of specified primitive type.
For example, all gpu convolution implementations should derive from typed_primitive_impl_ocl<convolution>.
*/
template <class PType>
struct typed_primitive_impl_ocl : public typed_primitive_impl<PType> {
    kernel_selector::kernel_data _kernel_data;
    std::vector<kernel::ptr> _kernels;

    // a pair of batch program hash and kernel entry hash of each ocl impl.
    std::pair<std::string, std::string> kernel_dump_info;

    typed_primitive_impl_ocl() : _kernel_data({}), _kernels({}) {}

    typed_primitive_impl_ocl(const typed_primitive_impl_ocl<PType>& other)
    : typed_primitive_impl<PType>(other._weights_reorder_params, other._kernel_name, other._is_dynamic)
    , _kernel_data(other._kernel_data)
    , _kernels({}) {
        _kernels.reserve(other._kernels.size());
        for (size_t k = 0; k < other._kernels.size(); ++k) {
            _kernels.emplace_back(other._kernels[k]->clone(other.can_share_kernels));
        }
        this->can_reuse_memory = _kernel_data.can_reuse_memory;
        this->can_share_kernels = other.can_share_kernels;
        this->m_manager = other.m_manager;
    }

    typed_primitive_impl_ocl(const kernel_selector::kernel_data& kd)
        : typed_primitive_impl<PType>(create_weights_reorder_params(kd.weightsReorderParams), kd.kernelName),
          _kernel_data(kd) {
        this->can_reuse_memory = _kernel_data.can_reuse_memory;
    }

    bool is_cpu() const override { return false; }

    // Cache blob format:
    //     [ kernel_selector::kernel_data ]
    //     [ kernel_ids ]
    void save(BinaryOutputBuffer& ob) const override {
        primitive_impl::save(ob);
        ob << make_data(&_kernel_data.internalBufferDataType, sizeof(kernel_selector::Datatype));
        ob << _kernel_data.internalBuffers;
        ob << _kernel_data.kernels;
        ob << _kernel_data.kernelName;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_impl::load(ib);
        ib >> make_data(&_kernel_data.internalBufferDataType, sizeof(kernel_selector::Datatype));
        ib >> _kernel_data.internalBuffers;
        ib >> _kernel_data.kernels;
        ib >> _kernel_data.kernelName;
    }

    template<typename ImplType>
    static std::unique_ptr<primitive_impl> create(const typed_program_node<PType>& arg, const kernel_impl_params& impl_param) {
        // concat buffer fusing for dynamic shape is adaptively applied at runtime. So we need to build dynamic impl at build time.
        if (impl_param.can_be_optimized() &&
            !((impl_param.is_type<concatenation>() ||
               impl_param.is_type<crop>() ||
               impl_param.runtime_skippable()) && impl_param.is_dynamic())) {
            return std::make_unique<ImplType>(kernel_selector::kernel_data{});
        }
        auto kernel_params = ImplType::get_kernel_params(ImplType::static_canonicalize_shapes(impl_param));
        kernel_params.is_shape_agnostic = impl_param.is_dynamic();
        kernel_params.set_dynamic_shape_offsets();
        auto& kernel_selector = ImplType::kernel_selector_t::Instance();
        auto best_kernel = kernel_selector.get_best_kernel(kernel_params);

        return std::make_unique<ImplType>(best_kernel);
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
        if (is_cpu()) {
            return;
        }

        _kernels.clear();
        if (!_kernel_data.kernels.empty()) {
            auto compiled_kernels = kernels_cache.get_kernels(params);
            _kernels.insert(_kernels.begin(), compiled_kernels.begin(), compiled_kernels.end());
            // batch program hash and kernel entry point to find corresponding cl source code
            kernel_dump_info = std::make_pair(std::to_string(kernels_cache.get_kernel_batch_hash(params)),
                                          _kernel_data.kernels[0].code.kernelString->entry_point);
            for (size_t i = 1; i < _kernel_data.kernels.size(); ++i)
                kernel_dump_info.second += " " + _kernel_data.kernels[i].code.kernelString->entry_point;
        }
        this->can_share_kernels = kernels_cache.get_kernels_reuse();
    }

    void init_by_cached_kernels(const kernels_cache& kernels_cache, std::vector<std::string>& cached_kernel_ids) override {
        if (is_cpu()) {
            return;
        }
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

    template<typename ImplType, typename KernelParamsType>
    static std::unique_ptr<primitive_impl> make_deep_copy(const ImplType& impl_ocl) {
        auto prim_impl = std::make_unique<ImplType>(impl_ocl);
        KernelParamsType* params_ptr = dynamic_cast<KernelParamsType*>((*prim_impl)._kernel_data.params.get());
        if (params_ptr != nullptr) {
            (*prim_impl)._kernel_data.params = std::make_unique<KernelParamsType>(*params_ptr);
        }
        return prim_impl;
    }

    std::vector<kernel::ptr> get_kernels() const override {
        return _kernels;
    }

    std::vector<BufferDescriptor> get_internal_buffer_descs(const kernel_impl_params&) const override {
        if (_kernel_data.internalBuffers.empty())
            return {};

        std::vector<BufferDescriptor> internal_buffers;
        auto dtype = from_data_type(_kernel_data.internalBufferDataType);
        const auto bpp = data_type_traits::size_of(dtype);
        for (const auto& buffer : _kernel_data.internalBuffers) {
            internal_buffers.emplace_back(buffer.byte_count / bpp, dtype, buffer.lockable);
        }
        return internal_buffers;
    }

    void set_arguments_impl(typed_primitive_inst<PType>& instance) override {
        if (instance.can_be_optimized() || is_cpu()) {
            return;
        }

        OPENVINO_ASSERT(_kernels.size() == _kernel_data.kernels.size(), "[GPU] Mismatch between compiled kernels count and expected kernels data\n",
                                                                        "[GPU] Compiled kernels count: ", _kernels.size(), "\n",
                                                                        "[GPU] KernelData count: ", _kernel_data.kernels.size(), "\n",
                                                                        "[GPU] Likely some issue with empty tensor handling happened");

        stream& stream = instance.get_network().get_stream();
        for (size_t kd_idx = 0; kd_idx < _kernel_data.kernels.size(); ++kd_idx) {
            if (_kernel_data.kernels[kd_idx].skip_execution) {
                continue;
            }

            auto args = get_arguments(instance);
            args.scalars = &_kernel_data.kernels[kd_idx].params.scalars;

            for (const auto& m : instance.get_intermediates_memories()) {
                args.intermediates.push_back(m);
            }

            stream.set_arguments(*_kernels[kd_idx], _kernel_data.kernels[kd_idx].params, args);
        }
    }

    void set_arguments_impl(typed_primitive_inst<PType>& instance, kernel_arguments_data& args) override {
        if (instance.can_be_optimized()) {
            return;
        }

        stream& stream = instance.get_network().get_stream();

        for (size_t k = 0; k < _kernels.size(); ++k) {
            if (_kernel_data.kernels[k].skip_execution)
                continue;

            stream.set_arguments(*_kernels[k], _kernel_data.kernels[k].params, args);
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
        OPENVINO_ASSERT(_kernels.size() == _kernel_data.kernels.size(), "[GPU] Mismatch between compiled kernels count and expected kernels data\n",
                                                                        "[GPU] Compiled kernels count: ", _kernels.size(), "\n",
                                                                        "[GPU] KernelData count: ", _kernel_data.kernels.size(), "\n",
                                                                        "[GPU] Likely some issue with empty tensor handling happened");
        for (size_t kd_idx = 0; kd_idx < _kernel_data.kernels.size(); ++kd_idx) {
            if (_kernel_data.kernels[kd_idx].skip_execution)
                continue;
            // If any user of the prim's users is CPU implementation or network's output, set prim as a output event (event won't be nullptr)
            bool needs_completion_event = instance.needs_completion_event();

            auto& params = _kernel_data.kernels[kd_idx].params;
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
            if (_kernel_data.needs_sub_kernels_sync) {
                tmp_events = {ev};
            }
            all_events.push_back(ev);
        }

        if ((all_events.size() == 0) && (tmp_events.size() > 0))
            return stream.aggregate_events(tmp_events);

        bool group_events = (all_events.size() > 1);
        return stream.aggregate_events(all_events, group_events);
    }

    std::vector<std::shared_ptr<cldnn::kernel_string>> get_kernels_source() override {
        std::vector<std::shared_ptr<cldnn::kernel_string>> kernel_strings;
        for (size_t i = 0; i < _kernel_data.kernels.size(); ++i) {
            kernel_strings.push_back(_kernel_data.kernels[i].code.kernelString);
        }
        return kernel_strings;
    }

    void reset_kernels_source() override {
        for (size_t i = 0; i < _kernel_data.kernels.size(); ++i) {
            _kernel_data.kernels[i].code.kernelString.reset();
        }
    }

    void set_kernels(cldnn::kernels_cache::compiled_kernels kernels) override {
        if (is_cpu())
            return;
        OPENVINO_ASSERT(kernels.size() == 1, "Only the kernels of the single primitive should be allowed.");
        auto& kernel_vec = kernels.begin()->second;
        _kernels.clear();
        _kernels.resize(kernel_vec.size());
        for (auto& k : kernel_vec) {
            auto sub_kernel_idx = k.second;
            _kernels[sub_kernel_idx] = k.first;
        }
    }

    std::pair<std::string, std::string> get_kernels_dump_info() const override {
        return kernel_dump_info;
    }

    virtual void update_dispatch_data(const kernel_impl_params& impl_params) {
        OPENVINO_ASSERT(this->_is_dynamic, "[GPU] update_dispatch_data() is called for static shape implementation ", this-> _kernel_name);
        OPENVINO_ASSERT(false, "[GPU] update_dispatch_data() is not implemented for dynamic implemenation ", this->_kernel_name);
    }
};

}  // namespace ocl
}  // namespace cldnn
