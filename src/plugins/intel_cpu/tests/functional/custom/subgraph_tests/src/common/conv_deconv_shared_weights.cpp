// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/runtime/properties.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

using ConvDeconvSharedWeightsParams = std::tuple<size_t,   // channels
                                                 size_t>;  // kernel size

/* Convolution and ConvolutionBackpropData consume the same weights Constant.
   Both nodes read the same buffer, but Deconvolution interprets it as a transposed (IO instead of OI) view.
   The repacked weights must not be shared between them through the weights cache,
   so the cache key has to take the source descriptor into account.
       Param0            Param1
         |                 |
    Convolution   ConvolutionBackpropData
          \           /
           \         /
            Weights (shared Constant [C, C, K, K])
*/
class ConvDeconvSharedWeightsTest : public testing::WithParamInterface<ConvDeconvSharedWeightsParams>,
                                    virtual public SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConvDeconvSharedWeightsParams>& obj) {
        const auto& [channels, kernel] = obj.param;
        std::ostringstream result;
        result << "C=" << channels << "_K=" << kernel;
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;
        configuration.insert({ov::hint::inference_precision.name(), ov::element::f32});
        const auto& [channels, kernel] = GetParam();

        const ov::Shape inputShape{1, channels, 8, 8};
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputShape),
                                   std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputShape)};

        // Square weights, so that the same Constant is a valid input for both Convolution ([O, I, K, K])
        // and ConvolutionBackpropData ([I, O, K, K]).
        const auto weights = ov::test::utils::make_constant(ov::element::f32, {channels, channels, kernel, kernel});

        const ov::Strides strides{1, 1};
        const ov::CoordinateDiff pads{0, 0};
        const ov::Strides dilations{1, 1};
        const auto conv = std::make_shared<ov::op::v1::Convolution>(params[0], weights, strides, pads, pads, dilations);
        const auto deconv =
            std::make_shared<ov::op::v1::ConvolutionBackpropData>(params[1], weights, strides, pads, pads, dilations);

        function = std::make_shared<ov::Model>(ov::OutputVector{conv, deconv}, params, "ConvDeconvSharedWeights");
    }
};

TEST_P(ConvDeconvSharedWeightsTest, CompareWithRefs) {
    run();
}

namespace {

INSTANTIATE_TEST_SUITE_P(smoke_ConvDeconvSharedWeights,
                         ConvDeconvSharedWeightsTest,
                         ::testing::Combine(::testing::Values(16, 32), ::testing::Values(1, 3)),
                         ConvDeconvSharedWeightsTest::getTestCaseName);

}  // namespace
}  // namespace test
}  // namespace ov
