// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/layer_test_utils.hpp"

namespace SubgraphTestsDefinitions {

enum class QuantizeType {
    FAKE_QUANTIZE,
    QUANTIZE_DEQUANTIZE,
    COMPRESSED_WEIGHTS,
    COMPRESSED_WEIGHTS_NO_SHIFT,
};

using QuantizedConvolutionBatchNormParams = std::tuple<QuantizeType, bool, std::string>;

class QuantizedConvolutionBatchNorm : public testing::WithParamInterface<QuantizedConvolutionBatchNormParams>,
                                            virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<QuantizedConvolutionBatchNormParams>& obj);

protected:
    void SetUp() override;
    void TearDown() override;
};

}  // namespace SubgraphTestsDefinitions
