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
#pragma once

#include "engine_impl.h"
#include "kernel_selector_common.h"
#include "kernel_runner_interface.h"
#include "kernel.h"
#include <vector>

namespace cldnn {
namespace gpu {

class kernel_runner : public kernel_selector::KernelRunnerInterface {
public:
    kernel_runner(engine_impl& engine_ref, uint32_t program_id, bool weights_and_bias_exist = false, bool zero_points_exist = false);

    std::vector<std::chrono::nanoseconds> run_kernels(const kernel_selector::KernelsData& kernelsData) override;

private:
    const int compilation_batch_size = 50;
    const int runs_per_kernel = 15;

    void prepare_kernel_args(const kernel_selector::KernelsData& kernels_data,
                             gpu::kernel::kernel_arguments_data& args);

    engine_impl::ptr engine;
    uint32_t program_id;
    bool weights_and_bias_exist;
    bool zero_points_exist;
    std::vector<memory_impl::cptr> input_buffers;
    std::vector<memory_impl::cptr> fused_ops_buffers;
    std::vector<memory_impl::ptr> output_buffers;
    std::vector<memory_impl::cptr> weight_buffers;
    std::vector<memory_impl::cptr> bias_buffers;
    std::vector<memory_impl::cptr> weight_zero_point_buffers;
    std::vector<memory_impl::cptr> activation_zero_point_buffers;
    std::vector<memory_impl::cptr> compensation_buffers;
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////
}  // namespace gpu
}  // namespace cldnn
