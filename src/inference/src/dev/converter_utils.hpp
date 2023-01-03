// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpp/ie_cnn_network.h"
#include "openvino/core/model.hpp"

namespace ov {
namespace legacy_convert {

InferenceEngine::CNNNetwork create_cnnnetwork(const std::shared_ptr<const ov::Model>& model, bool is_new_api);
void fill_input_info(const ov::Output<const ov::Node>& input, InferenceEngine::InputInfo::Ptr& inputInfo);
void fill_output_info(const ov::Output<const ov::Node>& output, InferenceEngine::DataPtr& outputInfo);

}  // namespace legacy_convert
}  // namespace ov
