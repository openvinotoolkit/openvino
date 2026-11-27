// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/node_builders/convolution.hpp"
#include "internal_properties.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/result.hpp"
#include "openvino/util/common_util.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

// Convolution -> [Relu] -> Multiply by a constant. Snippets are off: on static shapes they would take the Multiply
// before the CPU graph fusion (on dynamic shapes the fusion sees it either way). ACL builds used to fuse it with
// empty scales, and compile_model failed with "Check 'any_of(scale.size(), OC, 1U)'".
using ConvMultiplyParams = std::tuple<InputShape,  // input shape
                                      Shape,       // Multiply constant shape
                                      bool         // Relu before the Multiply
                                      >;

class ConvMultiplyCPUTest : public testing::WithParamInterface<ConvMultiplyParams>, virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConvMultiplyParams>& obj) {
        const auto& [inputShape, scaleShape, withRelu] = obj.param;
        std::ostringstream result;
        result << "IS=" << inputShape << "_ScaleShape=" << ov::util::vector_to_string(scaleShape)
               << "_withRelu=" << withRelu;
        return result.str();
    }

protected:
    void SetUp() override {
        const auto& [inputShape, scaleShape, withRelu] = GetParam();
        targetDevice = utils::DEVICE_CPU;
        configuration.insert(ov::hint::inference_precision(element::f32));
        configuration.insert(ov::intel_cpu::snippets_mode(ov::intel_cpu::SnippetsMode::DISABLE));
        init_input_shapes({inputShape});

        const auto param = std::make_shared<op::v0::Parameter>(element::f32, inputDynamicShapes[0]);
        auto output = utils::make_convolution(param,
                                              element::f32,
                                              {1, 1},
                                              {1, 1},
                                              {0, 0},
                                              {0, 0},
                                              {1, 1},
                                              op::PadType::EXPLICIT,
                                              8);
        if (withRelu) {
            output = std::make_shared<op::v0::Relu>(output);
        }
        output = std::make_shared<op::v1::Multiply>(output, utils::make_constant(element::f32, scaleShape));
        function = std::make_shared<Model>(ResultVector{std::make_shared<op::v0::Result>(output)},
                                           ParameterVector{param},
                                           "ConvMultiply");
    }
};

TEST_P(ConvMultiplyCPUTest, CompareWithRefs) {
    run();
}

namespace {
const std::vector<InputShape> perChannelShapes = {
    {{}, {{1, 16, 8, 8}}},
    {{-1, 16, -1, -1}, {{1, 16, 8, 8}, {2, 16, 5, 7}, {1, 16, 8, 8}}},
};

// H and W stay 8 so every target shape broadcasts with the [1, 1, 8, 8] scale
const std::vector<InputShape> spatialShapes = {
    {{}, {{1, 16, 8, 8}}},
    {{-1, 16, 8, 8}, {{1, 16, 8, 8}, {2, 16, 8, 8}}},
};

// The Multiply follows an already fused Relu
INSTANTIATE_TEST_SUITE_P(smoke_ConvMultiply_PerChannel,
                         ConvMultiplyCPUTest,
                         ::testing::Combine(::testing::ValuesIn(perChannelShapes),
                                            ::testing::Values(Shape{1, 8, 1, 1}),
                                            ::testing::Values(true)),
                         ConvMultiplyCPUTest::getTestCaseName);

// The Multiply is the first fused op; a spatial scale is not folded into the weights
INSTANTIATE_TEST_SUITE_P(smoke_ConvMultiply_Spatial,
                         ConvMultiplyCPUTest,
                         ::testing::Combine(::testing::ValuesIn(spatialShapes),
                                            ::testing::Values(Shape{1, 1, 8, 8}),
                                            ::testing::Values(false)),
                         ConvMultiplyCPUTest::getTestCaseName);
}  // namespace
}  // namespace test
}  // namespace ov
