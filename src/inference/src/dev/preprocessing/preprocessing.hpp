// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ie_input_info.hpp"
#include "openvino/pass/pass.hpp"

namespace ov {
namespace pass {

/**
 * @brief Converts the following preprocessing information to OpenVINO operations:
 *  - InferenceEngine::PreProcessInfo->PreProcessChannel::meanData -> Subtract
 *  - InferenceEngine::PreProcessInfo->PreProcessChannel::meanValue -> Subtract
 *  - InferenceEngine::PreProcessInfo->PreProcessChannel::stdScale -> Divide
 *
 * The order of operations is the following:
 *      (x - mean) / stdScale
 */
class AddPreprocessing : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("AddLegacyPreprocessing");
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};

}  // namespace pass
}  // namespace ov
