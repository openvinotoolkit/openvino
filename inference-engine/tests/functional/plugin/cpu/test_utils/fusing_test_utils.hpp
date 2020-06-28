// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu_test_utils.hpp"
#include "single_layer_tests/activation.hpp"

namespace FusingTestUtils {

typedef std::tuple<
        std::shared_ptr<ngraph::Function>,
        std::vector<std::string>> fusingSpecificParams;

std::shared_ptr<ngraph::Function> makeActivationPattern(std::vector<size_t> shape, ngraph::helpers::ActivationTypes type,
        double alpha = 0.0f, double beta = 0.0f);
std::shared_ptr<ngraph::Function> makeSwishPattern(std::vector<size_t> shape);
std::shared_ptr<ngraph::Function> makeActivationScaleShiftPattern(ngraph::helpers::ActivationTypes type, std::vector<size_t> shape);
std::shared_ptr<ngraph::Function> makeFakeQuantizeActivationPattern(size_t levels, ngraph::helpers::ActivationTypes type, std::vector<size_t> shape);
std::shared_ptr<ngraph::Function> makeSumPattern(std::vector<size_t> shape);
// todo: DWConvolution (not supported for GroupConvolution)

void inline CheckFusing(InferenceEngine::ExecutableNetwork &execNet, std::string nodeType, std::vector<std::string> fusedOps) {
    InferenceEngine::CNNNetwork execGraphInfo = execNet.GetExecGraphInfo();
    auto function = execGraphInfo.getFunction();
    ASSERT_NE(nullptr, function);
    for (const auto & op : function->get_ops()) {
        const auto & rtInfo = op->get_rt_info();

        auto getExecValue = [&rtInfo](const std::string & paramName) -> std::string {
            auto it = rtInfo.find(paramName);
            IE_ASSERT(rtInfo.end() != it);
            auto value = std::dynamic_pointer_cast<ngraph::VariantImpl<std::string>>(it->second);
            IE_ASSERT(nullptr != value);

            return value->get();
        };

        auto layerType = getExecValue("layerType");
        if (layerType == nodeType) {
            auto originalLayersNames = getExecValue("originalLayersNames");
            auto pos = originalLayersNames.find(nodeType);
            ASSERT_TRUE(pos != std::string::npos);
            for (auto fusedOp : fusedOps) {
                pos = originalLayersNames.find(fusedOp, pos);
                ASSERT_TRUE(pos != std::string::npos);
            }
        }
    }
}

} // namespace FusingTestUtils
