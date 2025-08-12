// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

enum class ConvType {
    CONVOLUTION,
    CONVOLUTION_BACKPROP,
};

enum class QuantizeType {
    FAKE_QUANTIZE,
    QUANTIZE_DEQUANTIZE,
    COMPRESSED_WEIGHTS,
    COMPRESSED_WEIGHTS_NO_SHIFT,
};

enum class IntervalsType {
    PER_TENSOR,
    PER_CHANNEL,
};

using QuantizedConvolutionBatchNormParams = std::tuple<ConvType, QuantizeType, IntervalsType, bool, std::string>;

class QuantizedConvolutionBatchNorm : public testing::WithParamInterface<QuantizedConvolutionBatchNormParams>,
                                      virtual public SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<QuantizedConvolutionBatchNormParams>& obj);

protected:
    void SetUp() override;
    void TearDown() override;
};

}  // namespace test
}  // namespace ov
