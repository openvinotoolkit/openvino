// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>
#include <string>
#include <memory>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ov_models/builders.hpp"

namespace LayerTestsDefinitions {
typedef std::tuple<
        std::vector<int64_t>,          // padsBegin
        std::vector<int64_t>,          // padsEnd
        float,                         // argPadValue
        ngraph::helpers::PadMode,      // padMode
        InferenceEngine::Precision,    // Net precision
        InferenceEngine::Precision,    // Input precision
        InferenceEngine::Precision,    // Output precision
        InferenceEngine::Layout,       // Input layout
        InferenceEngine::SizeVector,   // Input shapes
        LayerTestsUtils::TargetDevice  // Target device name
> padLayerTestParamsSet;

class PadLayerTest : public testing::WithParamInterface<padLayerTestParamsSet>,
                     virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<padLayerTestParamsSet>& obj);

protected:
    void SetUp() override;
    virtual std::shared_ptr<ov::Node> CreatePadOp(const ngraph::Output<ov::Node>& data,
                                      const std::vector<int64_t>& padsBegin,
                                      const std::vector<int64_t>& padsEnd,
                                      float argPadValue,
                                      ngraph::helpers::PadMode padMode) const {
        OPENVINO_SUPPRESS_DEPRECATED_START
        const auto pad = ngraph::builder::makePad(data, padsBegin, padsEnd, argPadValue, padMode, false);
        OPENVINO_SUPPRESS_DEPRECATED_END
        return pad;
    }
};

class PadLayerTest12 : public PadLayerTest {
protected:
    std::shared_ptr<ov::Node> CreatePadOp(const ngraph::Output<ov::Node>& data,
                                      const std::vector<int64_t>& padsBegin,
                                      const std::vector<int64_t>& padsEnd,
                                      float argPadValue,
                                      ngraph::helpers::PadMode padMode) const override {
        OPENVINO_SUPPRESS_DEPRECATED_START
        const auto pad = ngraph::builder::makePad(data, padsBegin, padsEnd, argPadValue, padMode, true);
        OPENVINO_SUPPRESS_DEPRECATED_END
        return pad;
    }
};
}  // namespace LayerTestsDefinitions
