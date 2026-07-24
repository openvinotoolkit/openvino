// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Copyright (C) 2026 FUJITSU LIMITED
//

#include "sycl_base_kernel.hpp"

#include <utility>
#include <vector>

namespace cldnn {
namespace sycl {

namespace {

memory::ptr resolve_arg_mem(const argument_desc& arg,
                            const kernel_arguments_data& data) {
    using T = argument_desc::Types;
    switch (arg.t) {
        case T::INPUT:
            OPENVINO_ASSERT(arg.index < data.inputs.size() && data.inputs[arg.index],
                            "The allocated input memory is necessary to set kernel arguments.");
            return std::const_pointer_cast<memory>(data.inputs[arg.index]);
        case T::OUTPUT:
            OPENVINO_ASSERT(arg.index < data.outputs.size() && data.outputs[arg.index],
                            "The allocated output memory is necessary to set kernel arguments.");
            return std::const_pointer_cast<memory>(data.outputs[arg.index]);
        case T::WEIGHTS:
            return std::const_pointer_cast<memory>(data.weights);
        case T::BIAS:
            return std::const_pointer_cast<memory>(data.bias);
        case T::SCALE_TABLE:
            return std::const_pointer_cast<memory>(data.scale_table);
        case T::SLOPE:
            return std::const_pointer_cast<memory>(data.slope);
        case T::INTERNAL_BUFFER:
            OPENVINO_ASSERT(arg.index < data.intermediates.size() && data.intermediates[arg.index],
                            "The allocated intermediate memory is necessary to set kernel arguments.");
            return std::const_pointer_cast<memory>(data.intermediates[arg.index]);
        case T::CELL:
            return std::const_pointer_cast<memory>(data.cell);
        case T::WEIGHTS_ZERO_POINTS:
            return std::const_pointer_cast<memory>(data.weights_zero_points);
        case T::ACTIVATIONS_ZERO_POINTS:
            return std::const_pointer_cast<memory>(data.activations_zero_points);
        case T::COMPENSATION:
            return std::const_pointer_cast<memory>(data.compensation);
        case T::INPUT_OF_FUSED_PRIMITIVE:
            OPENVINO_ASSERT(arg.index < data.fused_op_inputs.size() && data.fused_op_inputs[arg.index],
                            "The allocated fused_op_input memory is necessary to set kernel arguments.");
            return std::const_pointer_cast<memory>(data.fused_op_inputs[arg.index]);
        case T::SHAPE_INFO:
            return std::const_pointer_cast<memory>(data.shape_info);
        default:
            return nullptr;
    }
}

} // namespace

void sycl_base_kernel::set_arguments(const kernel_arguments_desc& args_desc,
                                     const kernel_arguments_data& data) {
    const auto& desc = args_desc.arguments;
    std::vector<arg_t> next;
    next.reserve(desc.size());

    for (const auto& ad : desc) {
        arg_t a;
        using T = argument_desc::Types;
        if (ad.t == T::SCALAR) {
            a.kind = arg_t::kind_t::SCALAR;
            OPENVINO_ASSERT(data.scalars && ad.index < data.scalars->size(),
                            "The allocated scalar is necessary to set kernel arguments.");
            a.scalar = (*data.scalars)[ad.index];
        } else if (ad.t == T::LOCAL_MEMORY_SIZE) {
            a.kind = arg_t::kind_t::LOCAL_MEM;
            OPENVINO_ASSERT(data.local_memory_args && ad.index < data.local_memory_args->size()
                            && (*data.local_memory_args)[ad.index],
                            "The allocated local memory is necessary to set kernel arguments.");
            a.local_size = (*data.local_memory_args)[ad.index];
        } else {
            a.kind = arg_t::kind_t::BUFFER;
            a.mem = resolve_arg_mem(ad, data);
        }
        next.push_back(std::move(a));
    }

    std::lock_guard<std::mutex> lock(_args_mutex);
    _stored_args = std::move(next);
}

std::vector<sycl_base_kernel::arg_t> sycl_base_kernel::stored_args() const {
    std::lock_guard<std::mutex> lock(_args_mutex);
    return _stored_args;
}

}  // namespace sycl
}  // namespace cldnn
