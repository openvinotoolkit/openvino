/*
// Copyright (c) 2016 Intel Corporation
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

#include <iterator>
#include "kernel.h"
#include "memory_gpu.h"

namespace cldnn { namespace gpu {

namespace {
    inline cl::NDRange toNDRange(const std::vector<size_t>& v)
    {
        switch (v.size())
        {
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

    void set_arguments(
        cl::Kernel& kernel,
        const kernel_selector::kernel_arguments& args,
        const kernel::kernel_arguments_data& data)
    {
        for (uint32_t i = 0; i < static_cast<uint32_t>(args.size()); i++)
        {
            cl_int status = CL_INVALID_ARG_VALUE;

            switch (args[i].t)
            {
            case kernel_selector::kernel_argument_types::INPUT:
                if (args[i].index < data.inputs.size() && data.inputs[args[i].index])
                {
                    const auto& input_mem = data.inputs[args[i].index];
                    if (input_mem)
                    {
                        status = kernel.setArg(i, dynamic_cast<const gpu::gpu_buffer&>(*input_mem).get_buffer());
                    }
                }
                break;
            case kernel_selector::kernel_argument_types::INTERNAL_BUFFER:
                if (args[i].index < data.intermediates.size() && data.intermediates[args[i].index])
                {
                    const auto& input_mem = data.intermediates[args[i].index];
                    if (input_mem)
                    {
                        status = kernel.setArg(i, dynamic_cast<const gpu::gpu_buffer&>(*input_mem).get_buffer());
                    }
                }
                break;
            case kernel_selector::kernel_argument_types::OUTPUT:
                if (data.output)
                {
                    if (data.output->get_layout().format.is_image_2d())
                        status = kernel.setArg(i, dynamic_cast<const gpu::gpu_image2d&>(*data.output).get_buffer());
                    else
                        status = kernel.setArg(i, dynamic_cast<const gpu::gpu_buffer&>(*data.output).get_buffer());
                }
                break;
            case kernel_selector::kernel_argument_types::WEIGHTS:
                if (data.weights)
                {
                    if (data.weights->get_layout().format.is_image_2d())
                        status = kernel.setArg(i, dynamic_cast<const gpu::gpu_image2d&>(*data.weights).get_buffer());
                    else
                        status = kernel.setArg(i, dynamic_cast<const gpu::gpu_buffer&>(*data.weights).get_buffer());
                }
                break;
            case kernel_selector::kernel_argument_types::BIAS:
                if (data.bias)
                {
                    status = kernel.setArg(i, dynamic_cast<const gpu::gpu_buffer&>(*data.bias).get_buffer());
                }
                break;
            case kernel_selector::kernel_argument_types::PREV_WEIGHTS_GRADIENT:
                if (data.prev_weights_grad)
                {
                    if (data.prev_weights_grad->get_layout().format.is_image_2d())
                        status = kernel.setArg(i, dynamic_cast<const gpu::gpu_image2d&>(*data.prev_weights_grad).get_buffer());
                    else
                        status = kernel.setArg(i, dynamic_cast<const gpu::gpu_buffer&>(*data.prev_weights_grad).get_buffer());
                }
                break;
            case kernel_selector::kernel_argument_types::PREV_BIAS_GRADIENT:
                if (data.prev_bias_grad)
                {
                    status = kernel.setArg(i, dynamic_cast<const gpu::gpu_buffer&>(*data.prev_bias_grad).get_buffer());
                }
                break;
            case kernel_selector::kernel_argument_types::WEIGHTS_QUANTIZATION_FACTORS:
                if (data.weights_quantization_factors)
                {
                    status = kernel.setArg(i, dynamic_cast<const gpu::gpu_buffer&>(*data.weights_quantization_factors).get_buffer());
                }
                break;
            case kernel_selector::kernel_argument_types::OUTPUT_CALIBRATION_FACTORS:
                if (data.output_calibration_factors)
                {
                    status = kernel.setArg(i, dynamic_cast<const gpu::gpu_buffer&>(*data.output_calibration_factors).get_buffer());
                }
                break;
            case kernel_selector::kernel_argument_types::SCALE_TABLE:
                if (data.scale_table)
                {
                    status = kernel.setArg(i, dynamic_cast<const gpu::gpu_buffer&>(*data.scale_table).get_buffer());
                }
                break;
            case kernel_selector::kernel_argument_types::SLOPE:
                if (data.slope)
                {
                    status = kernel.setArg(i, dynamic_cast<const gpu::gpu_buffer&>(*data.slope).get_buffer());
                }
                break;
            case kernel_selector::kernel_argument_types::SPLIT:
                status = kernel.setArg(i, data.split);
                break;
            case kernel_selector::kernel_argument_types::LEARNING_RATE:
                status = kernel.setArg(i, data.lr);
                break;
            case kernel_selector::kernel_argument_types::SCALAR:
                if (data.scalars && args[i].index < data.scalars->size())
                {
                    const auto& scalar = (*data.scalars)[args[i].index];
                    switch (scalar.t)
                    {
                    case kernel_selector::kernel_scalar_argument_types::UINT8:
                        status = kernel.setArg(i, scalar.v.u8);
                        break;
                    case kernel_selector::kernel_scalar_argument_types::UINT16:
                        status = kernel.setArg(i, scalar.v.u16);
                        break;
                    case kernel_selector::kernel_scalar_argument_types::UINT32:
                        status = kernel.setArg(i, scalar.v.u32);
                        break;
                    case kernel_selector::kernel_scalar_argument_types::UINT64:
                        status = kernel.setArg(i, scalar.v.u64);
                        break;
                    case kernel_selector::kernel_scalar_argument_types::INT8:
                        status = kernel.setArg(i, scalar.v.s8);
                        break;
                    case kernel_selector::kernel_scalar_argument_types::INT16:
                        status = kernel.setArg(i, scalar.v.s16);
                        break;
                    case kernel_selector::kernel_scalar_argument_types::INT32:
                        status = kernel.setArg(i, scalar.v.s32);
                        break;
                    case kernel_selector::kernel_scalar_argument_types::INT64:
                        status = kernel.setArg(i, scalar.v.s64);
                        break;
                    case kernel_selector::kernel_scalar_argument_types::FLOAT32:
                        status = kernel.setArg(i, scalar.v.f32);
                        break;
                    case kernel_selector::kernel_scalar_argument_types::FLOAT64:
                        status = kernel.setArg(i, scalar.v.f64);
                        break;
                    default:
                        break;
                    }
                }
                break;
            case kernel_selector::kernel_argument_types::RECURRENT: // RNN/LSTM/GRU layers
                if (data.recurrent)
                {
                    if (data.recurrent->get_layout().format.is_image_2d())
                        status = kernel.setArg(i, dynamic_cast<const gpu::gpu_image2d&>(*data.recurrent).get_buffer());
                    else
                        status = kernel.setArg(i, dynamic_cast<const gpu::gpu_buffer&>(*data.recurrent).get_buffer());
                }
                break;
            case kernel_selector::kernel_argument_types::HIDDEN: // RNN/LSTM/GRU layers
                if (data.hidden)
                {
                    if (data.hidden->get_layout().format.is_image_2d())
                        status = kernel.setArg(i, dynamic_cast<const gpu::gpu_image2d&>(*data.hidden).get_buffer());
                    else
                        status = kernel.setArg(i, dynamic_cast<const gpu::gpu_buffer&>(*data.hidden).get_buffer());
                }
                break;
            case kernel_selector::kernel_argument_types::CELL: // LSTMlayers
                if (data.cell)
                {
                    if (data.cell->get_layout().format.is_image_2d())
                        status = kernel.setArg(i, dynamic_cast<const gpu::gpu_image2d&>(*data.cell).get_buffer());
                    else
                        status = kernel.setArg(i, dynamic_cast<const gpu::gpu_buffer&>(*data.cell).get_buffer());
                }
                break;
            default:
                break;
            }

            if (status != CL_SUCCESS)
            {
                throw std::runtime_error("Error set args\n");
            }
        }
    }
}

event_impl::ptr kernel::run(
    const kernel_selector::cl_kernel_data& kernel_data,
    const std::vector<event_impl::ptr>& dependencies,
    const kernel_arguments_data& args) const
{
    auto clkernel = context()->get_kernels_cache().get_kernel(_kernel_id, _one_time_kernel);
    try {
        set_arguments(clkernel, kernel_data.arguments, args);
    }
    catch (cl::Error const& err) {
        throw ocl_error(err);
    }

    return context()->enqueue_kernel(clkernel, toNDRange(kernel_data.workGroups.global), toNDRange(kernel_data.workGroups.local), dependencies);
}

} }