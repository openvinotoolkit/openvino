// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/openvino.hpp>
#include <string>
#include <vector>

// clang-format off
#include "infer_request_wrap.hpp"
#include "utils.hpp"
// clang-format on

std::map<std::string, ov::runtime::TensorVector> getTensors(std::map<std::string, std::vector<std::string>> inputFiles,
                                                            std::vector<benchmark_app::InputsInfo>& app_inputs_info);

std::map<std::string, ov::runtime::TensorVector> getTensorsStaticCase(const std::vector<std::string>& inputFiles,
                                                                      const size_t& batchSize,
                                                                      benchmark_app::InputsInfo& app_inputs_info,
                                                                      size_t requestsNum);

void copyTensorData(ov::runtime::Tensor& dst, const ov::runtime::Tensor& src);
