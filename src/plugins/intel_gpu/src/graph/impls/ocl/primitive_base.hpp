// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <list>
#include <utility>
#include <vector>

#include "broadcast_inst.h"
#include "common_utils/kernel_selector_primitive_impl.hpp"
#include "concatenation_inst.h"
#include "gather_inst.h"
#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/graph/serialization/binary_buffer.hpp"
#include "intel_gpu/graph/serialization/cl_kernel_data_serializer.hpp"
#include "intel_gpu/graph/serialization/helpers.hpp"
#include "intel_gpu/graph/serialization/set_serializer.hpp"
#include "intel_gpu/graph/serialization/string_serializer.hpp"
#include "intel_gpu/graph/serialization/vector_serializer.hpp"
#include "kernel_selector_helper.h"
#include "permute_inst.h"
#include "primitive_inst.h"
#include "register.hpp"
#include "registry/implementation_map.hpp"
#include "scatter_elements_update_inst.h"
#include "scatter_nd_update_inst.h"
#include "scatter_update_inst.h"
#include "strided_slice_inst.h"

namespace cldnn {
namespace ocl {

inline void validate_f4e2m1_packed_output(const layout& output_layout, const char* primitive_name) {
    if (output_layout.data_type != ov::element::f4e2m1 || output_layout.is_dynamic()) {
        return;
    }

    OPENVINO_ASSERT(output_layout.get_linear_size() % 8 == 0,
                    "[GPU] ", primitive_name, ": f4e2m1 output size must be a multiple of 8 elements "
                    "(32-bit atomic write granularity), but got: ",
                    output_layout.get_linear_size());
}

/*
Base class for all GPU implementation of specified primitive type.
For example, all gpu convolution implementations should derive from typed_primitive_impl_ocl<convolution>.
*/
template <class PType>
struct typed_primitive_impl_ocl : public typed_primitive_impl_kernel_selector<PType> {
    using parent = typed_primitive_impl_kernel_selector<PType>;
    using parent::_execution_plan;
    using parent::_kernel_data;
    using parent::_kernels;
    using parent::get_arguments;
    using parent::rebuild_execution_plan;

    mutable KernelDumpInfo kernel_dump_info;

    typed_primitive_impl_ocl() = default;

    typed_primitive_impl_ocl(const typed_primitive_impl_ocl<PType>& other) : parent(other) {}

    typed_primitive_impl_ocl(const kernel_selector::kernel_data& kd) : parent(create_weights_reorder_params(kd.weightsReorderParams), kd) {}

    std::optional<format> get_preferred_input_format(size_t input_index) const override {
        const auto* params = dynamic_cast<const kernel_selector::base_params*>(_kernel_data.params.get());
        if (params == nullptr || input_index >= params->inputs.size()) {
            return std::nullopt;
        }
        return from_data_layout(params->inputs[input_index].GetLayout());
    }

    template<typename ImplType>
    static std::unique_ptr<primitive_impl> create(const typed_program_node<PType>& arg, const kernel_impl_params& impl_param) {
        // concat buffer fusing for dynamic shape is adaptively applied at runtime. So we need to build dynamic impl at build time.
        if (impl_param.can_be_optimized() &&
            ((!impl_param.is_type<concatenation>() &&
               !impl_param.is_type<crop>() &&
               !impl_param.runtime_skippable()) || !impl_param.is_dynamic())) {
            return std::make_unique<ImplType>(kernel_selector::kernel_data{});
        }
        auto kernel_params = ImplType::get_kernel_params(ImplType::static_canonicalize_shapes(impl_param));
        kernel_params.is_shape_agnostic = impl_param.is_dynamic();
        kernel_params.set_dynamic_shape_offsets();
        auto& kernel_selector = ImplType::kernel_selector_t::Instance();
        auto best_kernel = kernel_selector.get_best_kernel(kernel_params);

        return std::make_unique<ImplType>(best_kernel);
    }

protected:
    void init_kernels(const kernels_cache& kernels_cache, const kernel_impl_params& params) override {
        parent::init_kernels(kernels_cache, params);
        if (!this->_kernel_data.kernels.empty()) {
            kernel_dump_info.set_batch_hash(std::to_string(kernels_cache.get_kernel_batch_hash(params)));
        }
        this->can_share_kernels = kernels_cache.get_kernels_reuse();
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
        if (instance.can_be_optimized()) {
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
            args.local_memory_args = &_kernel_data.kernels[kd_idx].params.local_memory_args;

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

    event::ptr execute_impl(const std::vector<event::ptr>& events, typed_primitive_inst<PType>& instance) override {
        stream& stream = instance.get_network().get_stream();
        kernel_dump_info.clear_entries();
        if (instance.can_be_optimized()) {
            return stream.aggregate_events(events, events.size() > 1, instance.is_output());
        }
        OPENVINO_ASSERT(_kernels.size() == _kernel_data.kernels.size(), "[GPU] Mismatch between compiled kernels count and expected kernels data\n",
                                                                        "[GPU] Compiled kernels count: ", _kernels.size(), "\n",
                                                                        "[GPU] KernelData count: ", _kernel_data.kernels.size(), "\n",
                                                                        "[GPU] Likely some issue with empty tensor handling happened");
        return _execution_plan.execute(stream, _kernels, events, instance.needs_completion_event(), [&](size_t dispatch_index) {
            auto& params = _kernel_data.kernels[dispatch_index].params;
            auto args = get_arguments(instance);
            args.scalars = &params.scalars;
            for (const auto& memory : instance.get_intermediates_memories()) {
                args.intermediates.push_back(memory);
            }

            const auto& gws = params.workGroups.global;
            const auto& lws = params.workGroups.local;
            GPU_DEBUG_TRACE_DETAIL << "Enqueue kernel " << dispatch_index << ": gws=[" << gws[0] << ", " << gws[1] << ", " << gws[2] << "] lws=[" << lws[0]
                                   << ", " << lws[1] << ", " << lws[2] << "]" << (instance.needs_completion_event() ? " has_completion_event=true" : "")
                                   << std::endl;
            kernel_dump_info.add_entry_point(_kernels[dispatch_index]->get_id());
            return gpu_dispatch_binding{&params, std::move(args)};
        });
    }

    // Regardless of the model's dynamism, the compile time graph will rely on the skip_execution mechanism to determine which kernels will be executed
    // The runtime graph relies on the actual execution of the kernel in execute_impl(..)
    KernelDumpInfo get_kernels_dump_info(const cldnn::kernel_impl_params& impl_params) const override {
        if (kernel_dump_info.has_entries()) {
            return kernel_dump_info;
        }

        for (size_t i = 0; i < _kernel_data.kernels.size(); ++i) {
            if (_kernel_data.kernels[i].skip_execution) {
                continue;
            }

            if (_kernel_data.kernels[i].code.kernelString) {
                kernel_dump_info.add_entry_point(_kernel_data.kernels[i].code.kernelString->entry_point);
            } else if (i < _kernels.size() && _kernels[i]) {
                kernel_dump_info.add_entry_point(_kernels[i]->get_id());
            }
        }

        return kernel_dump_info;
    }
};

}  // namespace ocl
}  // namespace cldnn
