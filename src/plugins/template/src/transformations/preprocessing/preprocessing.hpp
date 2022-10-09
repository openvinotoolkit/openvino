// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/pass.hpp>

#include "ie_input_info.hpp"

namespace ngraph {
namespace pass {

class AddPreprocessing;

}  // namespace pass
}  // namespace ngraph

/**
 * @brief Converts the following preprocessing information to ngraph operations:
 *  - InferenceEngine::PreProcessInfo->PreProcessChannel::meanData -> Subtract
 *  - InferenceEngine::PreProcessInfo->PreProcessChannel::meanValue -> Subtract
 *  - InferenceEngine::PreProcessInfo->PreProcessChannel::stdScale -> Divide
 *
 * The order of operations is the following:
 *      (x - mean) / stdScale
 */
class ngraph::pass::AddPreprocessing : public ngraph::pass::FunctionPass {
    const InferenceEngine::InputsDataMap& m_inputInfoMap;

public:
    OPENVINO_RTTI("AddPreprocessing", "0");
    explicit AddPreprocessing(const InferenceEngine::InputsDataMap& inputInfoMap);

    bool run_on_model(const std::shared_ptr<ngraph::Function>& m) override;
};
