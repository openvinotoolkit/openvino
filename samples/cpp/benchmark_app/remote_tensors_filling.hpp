// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "utils.hpp"

namespace gpu {

using BufferType = void*;

std::map<std::string, ov::TensorVector> get_remote_input_tensors(
    const std::map<std::string, std::vector<std::string>>& inputFiles,
    const std::vector<benchmark_app::InputsInfo>& app_inputs_info,
    const ov::CompiledModel& compiledModel,
    std::vector<BufferType>& clBuffer,
    size_t num_requests);

std::map<std::string, ov::Tensor> get_remote_output_tensors(const ov::CompiledModel& compiledModel,
                                                            std::map<std::string, ::gpu::BufferType>& clBuffer);
}  // namespace gpu

namespace npu {

std::map<std::string, ov::TensorVector> get_remote_input_tensors(
    const std::map<std::string, std::vector<std::string>>& inputFiles,
    const std::vector<benchmark_app::InputsInfo>& app_inputs_info,
    const ov::CompiledModel& compiledModel,
    size_t num_requests);
}  // namespace npu
