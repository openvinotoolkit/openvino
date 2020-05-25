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

#include "kernel_runner.h"
#include "kernel.h"
#include "weight_bias_params.h"
#include <chrono>
#include <vector>
#include <limits>
#include <algorithm>

namespace cldnn {
namespace gpu {

kernel_runner::kernel_runner(engine_impl& engine_ref, uint32_t program_id, bool weights_and_bias_exist, bool zero_points_exist)
    : engine(&engine_ref), program_id(program_id), weights_and_bias_exist(weights_and_bias_exist), zero_points_exist(zero_points_exist) {}

void kernel_runner::prepare_kernel_args(const kernel_selector::KernelsData& kernels_data,
                                        gpu::kernel::kernel_arguments_data& args) {
    const auto& base_params = *static_cast<kernel_selector::base_params*>(kernels_data[0].params.get());
    // Prepare input buffers
    if (input_buffers.empty()) {
        for (const auto& input : base_params.inputs) {
            int num_of_input_elements = static_cast<int>(input.PhysicalSize());
            input_buffers.push_back(engine->allocate_memory(
                {from_data_type(input.GetDType()), format::bfyx, tensor(1, 1, num_of_input_elements, 1)},
                0));
        }
    }
    for (const auto& input : input_buffers) {
        args.inputs.push_back(input);
    }
    // Prepare fused operations buffers
    if (fused_ops_buffers.empty()) {
        for (auto& fused_op : base_params.fused_ops) {
            for (auto& fused_ops_input : fused_op.tensors) {
                auto num_of_elements = static_cast<int>(fused_ops_input.PhysicalSize());
                fused_ops_buffers.push_back(engine->allocate_memory(
                    { from_data_type(fused_ops_input.GetDType()), format::bfyx, tensor(1, 1, num_of_elements, 1) },
                    0));
            }
        }
    }
    for (const auto& fused_op_input : fused_ops_buffers) {
        args.fused_op_inputs.push_back(fused_op_input);
    }
    // Prepare output buffer
    if (output_buffers.empty()) {
        int num_of_output_elements = static_cast<int>(base_params.output.PhysicalSize());
        output_buffers.push_back(engine->allocate_memory(
            {from_data_type(base_params.output.GetDType()), format::bfyx, tensor(1, 1, num_of_output_elements, 1)},
            0));
    }

    args.output = output_buffers[0];

    if (weights_and_bias_exist) {
        // Prepare weight buffer
        const auto& weights_bias_params =
            *static_cast<kernel_selector::weight_bias_params*>(kernels_data[0].params.get());
        int num_of_weight_elements_ifm = static_cast<int>(weights_bias_params.weights.IFM().v);
        int num_of_weight_elements_spatial_y = static_cast<int>(weights_bias_params.weights.Y().v);
        int num_of_weight_elements_spatial_x = static_cast<int>(weights_bias_params.weights.X().v);
        int num_of_weight_elements_spatial = static_cast<int>(weights_bias_params.weights.PhysicalSize());
        int num_of_weight_elements_ofm = 1;

        cldnn::format::type fmt = cldnn::format::bfyx;

        if (!cldnn::format::is_image_2d(from_weights_layout(weights_bias_params.weights.GetLayout()))) {
            if (weight_buffers.empty())
                weight_buffers.push_back(
                    engine->allocate_memory({from_weights_type(weights_bias_params.weights.GetDType()),
                                             fmt,
                                             tensor(num_of_weight_elements_ofm, 1, num_of_weight_elements_spatial, 1)},
                                            0));

            if (weight_buffers[0]->get_layout().format != fmt)
                weight_buffers[0] =
                    engine->allocate_memory({from_weights_type(weights_bias_params.weights.GetDType()),
                                             fmt,
                                             tensor(num_of_weight_elements_ofm, 1, num_of_weight_elements_spatial, 1)},
                                            0);

            while (weight_buffers[0]->get_layout().bytes_count() < weights_bias_params.weights.PhysicalSizeInBytes()) {
                // Weights layout depends on the kernel. Multiply the buffer size by 2 until it is big enough
                // (to avoid complex computations of the exact buffer size according to the chosen layout).
                weight_buffers.clear();
                num_of_weight_elements_spatial *= 2;
                weight_buffers.push_back(
                    engine->allocate_memory({from_weights_type(weights_bias_params.weights.GetDType()),
                                             fmt,
                                             tensor(num_of_weight_elements_ofm, 1, num_of_weight_elements_spatial, 1)},
                                            0));
            }
        } else {
            weight_buffers.clear();
            fmt = from_weights_layout(weights_bias_params.weights.GetLayout());
            num_of_weight_elements_ofm = static_cast<int>(weights_bias_params.weights.OFM().v);
            weight_buffers.push_back(engine->allocate_memory({from_weights_type(weights_bias_params.weights.GetDType()),
                                                              fmt,
                                                              tensor(num_of_weight_elements_ofm,
                                                                     num_of_weight_elements_ifm,
                                                                     num_of_weight_elements_spatial_x,
                                                                     num_of_weight_elements_spatial_y)},
                                                             0));
        }
        args.weights = weight_buffers[0];

        // Prepare bias buffer
        if (!weights_bias_params.bias.empty()) {
            if (bias_buffers.empty()) {
                int num_of_bias_elements = static_cast<int>(weights_bias_params.bias[0].PhysicalSize());
                bias_buffers.push_back(engine->allocate_memory({from_data_type(weights_bias_params.bias[0].GetDType()),
                                                                format::bfyx,
                                                                tensor(1, num_of_bias_elements, 1, 1)},
                                                               0));
            }
            args.bias = bias_buffers[0];
        }
        if (zero_points_exist) {
            const auto& zero_point_params =
                static_cast<const kernel_selector::weight_bias_zero_point_params&>(weights_bias_params);
            if (weight_zero_point_buffers.empty()) {
                for (auto& weight_zero_point : zero_point_params.weights_zero_points) {
                    auto num_of_elements = static_cast<int>(weight_zero_point.PhysicalSize());
                    weight_zero_point_buffers.push_back(
                        engine->allocate_memory({
                            from_data_type(weight_zero_point.GetDType()),
                            format::bfyx,
                            tensor(1, num_of_elements, 1, 1) },
                            0));
                }
            }
            if (activation_zero_point_buffers.empty()) {
                for (auto& activation_zero_point : zero_point_params.activations_zero_points) {
                    auto num_of_elements = static_cast<int>(activation_zero_point.PhysicalSize());
                    weight_zero_point_buffers.push_back(
                        engine->allocate_memory({
                            from_data_type(activation_zero_point.GetDType()),
                            format::bfyx,
                            tensor(1, num_of_elements, 1, 1) },
                            0));
                }
            }
            if (compensation_buffers.empty()) {
                for (auto& compensation : zero_point_params.compensation) {
                    auto num_of_elements = static_cast<int>(compensation.PhysicalSize());
                    weight_zero_point_buffers.push_back(
                        engine->allocate_memory({
                            from_data_type(compensation.GetDType()),
                            format::bfyx,
                            tensor(1, num_of_elements, 1, 1) },
                            0));
                }
            }
        }
    }

    args.split = 0;
}

std::vector<std::chrono::nanoseconds> kernel_runner::run_kernels(const kernel_selector::KernelsData& kernels_data) {
    auto context = engine->get_context();

    std::vector<std::chrono::nanoseconds> run_times;

    int num_of_kernels_to_run = static_cast<int>(kernels_data.size());
    int num_of_kernels_run = 0;

    kernel_selector::KernelsData::const_iterator batch_start = kernels_data.begin();
    kernel_selector::KernelsData::const_iterator batch_end;
    while (num_of_kernels_to_run > 0) {
        int current_compilation_batch = std::min(num_of_kernels_to_run, compilation_batch_size);
        batch_end = batch_start + current_compilation_batch;

        std::vector<gpu::kernel> kernels;
        std::vector<std::chrono::nanoseconds> kernel_run_times;

        for (auto it = batch_start; it < batch_end; it++) {
            kernels.push_back(kernel(context, it->kernels[0].kernelString, program_id, false, true));
        }

        gpu::kernel::kernel_arguments_data args;

        prepare_kernel_args(kernels_data, args);
        context->queue(0).finish();

        int i = 0;
        for (auto it = batch_start; it < batch_end; it++) {
            std::vector<event_impl::ptr> events;
            int num_of_runs = 0;

            for (int iteration = 0; iteration < runs_per_kernel; iteration++) {
                event_impl::ptr event;
                try {
                    event = kernels[i].run(0, it->kernels[0], {}, args);
                } catch (...) {
                    // Could not run this kernel. Push back NULL event (will be ignored later).
                }
                events.push_back(event);
            }

            context->queue(0).finish();

            for (auto& event : events) {
                if (event.get() != NULL) {
                    auto profiling_intervals = event->get_profiling_info();
                    for (auto const& profiling_interval : profiling_intervals) {
                        if (profiling_interval.name == "executing") {
                            kernel_run_times.push_back(profiling_interval.value->value());
                            num_of_runs++;
                            break;
                        }
                    }
                }
            }

            if (num_of_runs > 0) {
                std::sort(kernel_run_times.begin(), kernel_run_times.end());
                run_times.push_back(kernel_run_times[0]);
                num_of_kernels_run += 1;
            } else {
                run_times.push_back(std::chrono::nanoseconds::max());
            }
            kernel_run_times.clear();
            i++;
        }

        num_of_kernels_to_run -= current_compilation_batch;
        batch_start += current_compilation_batch;
    }

    if (num_of_kernels_run == 0) {
        // If all kernels failed to run throw to avoid corrupting cache
        throw std::runtime_error("kernel_runner::run_kernels - could not run any of provided kernels");
    }

    return run_times;
}

}  // namespace gpu
}  // namespace cldnn
