// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils/cpu_test_utils.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ov_models/utils/ov_helpers.hpp"
#include "ov_models/builders.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;

namespace LayerTestsDefinitions {

class AddConvertToReorderTest : virtual public LayerTestsUtils::LayerTestsCommon {
public:
    void BuildGraph(const ngraph::element::Type& secondInpType) {
        secondConstantType = secondInpType;
        int axis = 2;
        std::vector<int> indices = {0, 3, 2, 1};
        std::vector<size_t> indicesShape = {2, 2};
        std::vector<size_t> inputShape = {10, 20, 30, 40};

        InferenceEngine::Precision netPrecision = inPrc = outPrc = Precision::FP32;
        targetDevice = ov::test::utils::DEVICE_CPU;

        ASSERT_EQ(ngraph::shape_size(indicesShape), indices.size())
                                    << "Indices vector size and provided indices shape doesn't fit each other";
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};
        auto paramOuts = ngraph::helpers::convert2OutputVector(
                ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
        auto indicesNode = ngraph::opset3::Constant::create(secondConstantType, ngraph::Shape(indicesShape), indices);
        auto axisNode = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape({}), {axis});
        auto gather = std::make_shared<ngraph::opset3::Gather>(paramOuts[0], indicesNode, axisNode);
        ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(gather)};
        function = std::make_shared<ngraph::Function>(results, params, "gather");
    }
    std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>> CalculateRefs() override {
        // Convert the second input constant precision to i64 to run the reference function
        if (ngraph::element::Type_t::i8 == secondConstantType) {
            ngraph::pass::ConvertPrecision<ngraph::element::Type_t::i8, ngraph::element::Type_t::i64>().run_on_model(functionRefs);
        } else if (ngraph::element::Type_t::bf16 == secondConstantType) {
            ngraph::pass::ConvertPrecision<ngraph::element::Type_t::bf16, ngraph::element::Type_t::i64>().run_on_model(functionRefs);
        }
        return LayerTestsUtils::LayerTestsCommon::CalculateRefs();
    }

private:
    ngraph::element::Type secondConstantType;
};

namespace  {

/* Test insertion of the Reorder layer if there is one.

    Parameter[FP32]     Constant[I8]
          \                 /
           \               /
            \       Reorder[I32] (Is inserted by the Graph)
             \           /
             Gather[FP32]
                  |
                  |
             Output[FP32]
*/
TEST_F(AddConvertToReorderTest, smoke_TestAddReorder_CPU) {
    BuildGraph(ngraph::element::i8);
    Run();
    CheckNumberOfNodesWithType(executableNetwork, "Convert", 0);
    CheckNumberOfNodesWithType(executableNetwork, "Reorder", 1);
}
} // namespace
} // namespace LayerTestsDefinitions
