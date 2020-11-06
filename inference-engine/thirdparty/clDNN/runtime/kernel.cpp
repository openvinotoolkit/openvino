/*
// Copyright (c) 2016-2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "cldnn/runtime/refcounted_obj.h"

#include "kernel.h"
#include "memory_gpu.h"
#include "memory_impl.h"
#include <vector>
#include <iterator>

namespace cldnn {
namespace gpu {

namespace {
inline cl::NDRange toNDRange(const std::vector<size_t>& v) {
    switch (v.size()) {
        case 1:
            return cl::NDRange(v[0]);
        case 2:
            return cl::NDRange(v[0], v[1]);
        case 3:
            return cl::NDRange(v[0], v[1], v[2]);
        default:
            return cl::NullRange;
    }
}

void set_arguments_impl(kernel_type& kernel,
                        const arguments_desc& args,
                        const kernel_arguments_data& data) {
    using args_t = argument_desc::Types;
    using scalar_t = scalar_desc::Types;
    for (uint32_t i = 0; i < static_cast<uint32_t>(args.size()); i++) {
        cl_int status = CL_INVALID_ARG_VALUE;
        switch (args[i].t) {
            case args_t::INPUT:
                if (args[i].index < data.inputs.size() && data.inputs[args[i].index]) {
                    const auto& input_mem = data.inputs[args[i].index];
                    if (input_mem) {
                        if (input_mem->get_layout().format.is_image_2d())
                            status = kernel.setArg(i, dynamic_cast<const gpu::gpu_image2d&>(*input_mem).get_buffer());
                        else if (memory_capabilities::is_usm_type(input_mem->get_allocation_type()))
                            status = kernel.setArgUsm(i, dynamic_cast<const gpu::gpu_usm&>(*input_mem).get_buffer());
                        else
                            status = kernel.setArg(i, dynamic_cast<const gpu::gpu_buffer&>(*input_mem).get_buffer());
                    }
                }
                break;
            case args_t::INPUT_OF_FUSED_PRIMITIVE:
                if (args[i].index < data.fused_op_inputs.size() && data.fused_op_inputs[args[i].index]) {
                    const auto& input_mem = data.fused_op_inputs[args[i].index];
                    if (input_mem) {
                        if (memory_capabilities::is_usm_type(input_mem->get_allocation_type()))
                            status = kernel.setArgUsm(i, dynamic_cast<const gpu::gpu_usm&>(*input_mem).get_buffer());
                        else
                            status = kernel.setArg(i, dynamic_cast<const gpu::gpu_buffer&>(*input_mem).get_buffer());
                    }
                }
                break;
            case args_t::INTERNAL_BUFFER:
                if (args[i].index < data.intermediates.size() && data.intermediates[args[i].index]) {
                    const auto& input_mem = data.intermediates[args[i].index];
                    if (input_mem) {
                        if (memory_capabilities::is_usm_type(input_mem->get_allocation_type()))
                            status = kernel.setArgUsm(i, dynamic_cast<const gpu::gpu_usm&>(*input_mem).get_buffer());
                        else
                            status = kernel.setArg(i, dynamic_cast<const gpu::gpu_buffer&>(*input_mem).get_buffer());
                    }
                }
                break;
            case args_t::OUTPUT:
                if (data.output) {
                     if (data.output->get_layout().format.is_image_2d())
                        status = kernel.setArg(i, dynamic_cast<const gpu::gpu_image2d&>(*data.output).get_buffer());
                     else if (memory_capabilities::is_usm_type(data.output->get_allocation_type()))
                         status = kernel.setArgUsm(i, dynamic_cast<const gpu::gpu_usm&>(*data.output).get_buffer());
                     else
                        status = kernel.setArg(i, dynamic_cast<const gpu::gpu_buffer&>(*data.output).get_buffer());
                }
                break;
            case args_t::WEIGHTS:
                if (data.weights) {
                    if (data.weights->get_layout().format.is_image_2d())
                        status = kernel.setArg(i, dynamic_cast<const gpu::gpu_image2d&>(*data.weights).get_buffer());
                    else if (memory_capabilities::is_usm_type(data.weights->get_allocation_type()))
                        status = kernel.setArgUsm(i, dynamic_cast<const gpu::gpu_usm&>(*data.weights).get_buffer());
                    else
                        status = kernel.setArg(i, dynamic_cast<const gpu::gpu_buffer&>(*data.weights).get_buffer());
                }
                break;
            case args_t::BIAS:
                if (data.bias) {
                    if (memory_capabilities::is_usm_type(data.bias->get_allocation_type()))
                        status = kernel.setArgUsm(i, dynamic_cast<const gpu::gpu_usm&>(*data.bias).get_buffer());
                    else
                        status = kernel.setArg(i, dynamic_cast<const gpu::gpu_buffer&>(*data.bias).get_buffer());
                }
                break;
            case args_t::WEIGHTS_ZERO_POINTS:
                if (data.weights_zero_points) {
                    if (memory_capabilities::is_usm_type(data.weights_zero_points->get_allocation_type()))
                        status = kernel.setArgUsm(
                            i,
                            dynamic_cast<const gpu::gpu_usm&>(*data.weights_zero_points).get_buffer());
                    else
                        status = kernel.setArg(
                            i,
                            dynamic_cast<const gpu::gpu_buffer&>(*data.weights_zero_points).get_buffer());
                }
                break;
            case args_t::ACTIVATIONS_ZERO_POINTS:
                if (data.activations_zero_points) {
                    if (memory_capabilities::is_usm_type(data.activations_zero_points->get_allocation_type()))
                        status = kernel.setArgUsm(
                            i,
                            dynamic_cast<const gpu::gpu_usm&>(*data.activations_zero_points).get_buffer());
                    else
                        status = kernel.setArg(
                            i,
                            dynamic_cast<const gpu::gpu_buffer&>(*data.activations_zero_points).get_buffer());
                }
                break;
            case args_t::COMPENSATION:
                if (data.compensation) {
                    if (memory_capabilities::is_usm_type(data.compensation->get_allocation_type()))
                        status = kernel.setArgUsm(
                                i,
                                dynamic_cast<const gpu::gpu_usm&>(*data.compensation).get_buffer());
                    else
                        status = kernel.setArg(
                                 i,
                                 dynamic_cast<const gpu::gpu_buffer&>(*data.compensation).get_buffer());
                }
                break;
            case args_t::SCALE_TABLE:
                if (data.scale_table) {
                    if (memory_capabilities::is_usm_type(data.scale_table->get_allocation_type()))
                        status = kernel.setArgUsm(i, dynamic_cast<const gpu::gpu_usm&>(*data.scale_table).get_buffer());
                    else
                        status = kernel.setArg(i, dynamic_cast<const gpu::gpu_buffer&>(*data.scale_table).get_buffer());
                }
                break;
            case args_t::SLOPE:
                if (data.slope) {
                    if (memory_capabilities::is_usm_type(data.slope->get_allocation_type()))
                        status = kernel.setArgUsm(i, dynamic_cast<const gpu::gpu_usm&>(*data.slope).get_buffer());
                    else
                        status = kernel.setArg(i, dynamic_cast<const gpu::gpu_buffer&>(*data.slope).get_buffer());
                }
                break;
            case args_t::SPLIT:
                status = kernel.setArg(i, data.split);
                break;
            case args_t::SCALAR:
                if (data.scalars && args[i].index < data.scalars->size()) {
                    const auto& scalar = (*data.scalars)[args[i].index];
                    switch (scalar.t) {
                        case scalar_t::UINT8:
                            status = kernel.setArg(i, scalar.v.u8);
                            break;
                        case scalar_t::UINT16:
                            status = kernel.setArg(i, scalar.v.u16);
                            break;
                        case scalar_t::UINT32:
                            status = kernel.setArg(i, scalar.v.u32);
                            break;
                        case scalar_t::UINT64:
                            status = kernel.setArg(i, scalar.v.u64);
                            break;
                        case scalar_t::INT8:
                            status = kernel.setArg(i, scalar.v.s8);
                            break;
                        case scalar_t::INT16:
                            status = kernel.setArg(i, scalar.v.s16);
                            break;
                        case scalar_t::INT32:
                            status = kernel.setArg(i, scalar.v.s32);
                            break;
                        case scalar_t::INT64:
                            status = kernel.setArg(i, scalar.v.s64);
                            break;
                        case scalar_t::FLOAT32:
                            status = kernel.setArg(i, scalar.v.f32);
                            break;
                        case scalar_t::FLOAT64:
                            status = kernel.setArg(i, scalar.v.f64);
                            break;
                        default:
                            break;
                    }
                }
                break;
            case args_t::RECURRENT:  // RNN/LSTM/GRU layers
                if (data.recurrent) {
                    if (data.recurrent->get_layout().format.is_image_2d())
                        status = kernel.setArg(i, dynamic_cast<const gpu::gpu_image2d&>(*data.recurrent).get_buffer());
                    else if (memory_capabilities::is_usm_type(data.recurrent->get_allocation_type()))
                        status = kernel.setArgUsm(i, dynamic_cast<const gpu::gpu_usm&>(*data.recurrent).get_buffer());
                    else
                        status = kernel.setArg(i, dynamic_cast<const gpu::gpu_buffer&>(*data.recurrent).get_buffer());
                }
                break;
            case args_t::HIDDEN:  // RNN/LSTM/GRU layers
                if (data.hidden) {
                    if (data.hidden->get_layout().format.is_image_2d())
                        status = kernel.setArg(i, dynamic_cast<const gpu::gpu_image2d&>(*data.hidden).get_buffer());
                    else if (memory_capabilities::is_usm_type(data.hidden->get_allocation_type()))
                        status = kernel.setArgUsm(i, dynamic_cast<const gpu::gpu_usm&>(*data.hidden).get_buffer());
                    else
                        status = kernel.setArg(i, dynamic_cast<const gpu::gpu_buffer&>(*data.hidden).get_buffer());
                }
                break;
            case args_t::CELL:  // LSTMlayers
                if (data.cell) {
                    if (data.cell->get_layout().format.is_image_2d())
                        status = kernel.setArg(i, dynamic_cast<const gpu::gpu_image2d&>(*data.cell).get_buffer());
                    else if (memory_capabilities::is_usm_type(data.cell->get_allocation_type()))
                        status = kernel.setArgUsm(i, dynamic_cast<const gpu::gpu_usm&>(*data.cell).get_buffer());
                    else
                        status = kernel.setArg(i, dynamic_cast<const gpu::gpu_buffer&>(*data.cell).get_buffer());
                }
                break;
            default:
                break;
        }

        if (status != CL_SUCCESS) {
            throw std::runtime_error("Error set arg " + std::to_string(i) + ", error code: " + std::to_string(status) + "\n");
        }
    }
}
}  // namespace

void kernel::set_arguments(uint32_t queue_id,
                           const kernel_arguments_desc& args_desc,
                           const kernel_arguments_data& args) {
    static std::mutex m;
    std::lock_guard<std::mutex> guard(m);

    // Create a copy of cl kernel for each stream if it doesn't exist
    // Copy is needed to avoid data races between streams, but we create it only once for each stream
    // because the cloning is quite expensive.
    // Mutex is still needed to ensure that insert operation into the map is thread safe
    if (_cl_kernels.find(queue_id) == _cl_kernels.end())
        _cl_kernels[queue_id] = _compiled_kernel.clone();

    try {
        set_arguments_impl(_cl_kernels.at(queue_id), args_desc.arguments, args);
    } catch (cl::Error const& err) {
        throw ocl_error(err);
    }
}

void kernel::cleanup(uint32_t queue_id) {
    _cl_kernels.erase(queue_id);
}

event_impl::ptr kernel::run(uint32_t queue_id,
                            const kernel_arguments_desc& args_desc,
                            const std::vector<event_impl::ptr>& dependencies) const {

    if (_cl_kernels.find(queue_id) == _cl_kernels.end() || _cl_kernels.at(queue_id).get() == NULL) {
        throw std::runtime_error("[clDNN] Kernel for layer " + args_desc.layerID + " is not found for stream " + std::to_string(queue_id));
    }

    return context()->enqueue_kernel(queue_id,
                                     _cl_kernels.at(queue_id),
                                     toNDRange(args_desc.workGroups.global),
                                     toNDRange(args_desc.workGroups.local),
                                     dependencies);
}

}  // namespace gpu
}  // namespace cldnn
