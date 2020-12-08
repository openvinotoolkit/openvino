// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils/cpu_test_utils.hpp"
#include "functional_test_utils/layer_test_utils.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"

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
        targetDevice = CommonTestUtils::DEVICE_CPU;

        ASSERT_EQ(ngraph::shape_size(indicesShape), indices.size())
                                    << "Indices vector size and provided indices shape doesn't fit each other";
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
        auto paramOuts = ngraph::helpers::convert2OutputVector(
                ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
        auto indicesNode = ngraph::opset3::Constant::create(secondConstantType, ngraph::Shape(indicesShape), indices);
        auto axisNode = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape({}), {axis});
        auto gather = std::make_shared<ngraph::opset3::Gather>(paramOuts[0], indicesNode, axisNode);
        ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(gather)};
        function = std::make_shared<ngraph::Function>(results, params, "gather");
    }
    std::vector<std::vector<std::uint8_t>> CalculateRefs() override {
        // Convert the second input constant precision to i64 to run the reference function
        switch (secondConstantType) {
            case ngraph::element::Type_t::i8:
                ngraph::pass::ConvertPrecision<ngraph::element::Type_t::i8, ngraph::element::Type_t::i64>().run_on_function(function);
                break;
            case ngraph::element::Type_t::bf16:
                ngraph::pass::ConvertPrecision<ngraph::element::Type_t::bf16, ngraph::element::Type_t::i64>().run_on_function(function);
                break;
            default:
                // pass
                break;
        }

        return LayerTestsUtils::LayerTestsCommon::CalculateRefs();
    }
    void CheckElementOfTypeCount(std::string typeName, size_t expectedCount) {
        InferenceEngine::CNNNetwork execGraphInfo = executableNetwork.GetExecGraphInfo();
        auto function = execGraphInfo.getFunction();
        ASSERT_NE(nullptr, function);
        size_t actualPermuteCount = 0;
        for (const auto &node : function->get_ops()) {
            const auto & rtInfo = node->get_rt_info();
            auto getExecValue = [&rtInfo](const std::string & paramName) -> std::string {
                auto it = rtInfo.find(paramName);
                IE_ASSERT(rtInfo.end() != it);
                auto value = std::dynamic_pointer_cast<ngraph::VariantImpl<std::string>>(it->second);
                IE_ASSERT(nullptr != value);
                return value->get();
            };
            if (getExecValue(ExecGraphInfoSerialization::LAYER_TYPE) == typeName) {
                actualPermuteCount++;
            }
        }

        ASSERT_EQ(expectedCount, actualPermuteCount) << "Unexpected count of the element type '" << typeName << "' ";
    }

private:
    ngraph::element::Type secondConstantType;
};

namespace  {
/* Test insertion of the Convert layer if there is no suitable reorder.

    Parameter[FP32]     Constant[BF16]
          \                 /
           \               /
            \       Convert[I32] (Is inserted by the MKLDNNGraphOptimizer)
             \           /
             Gather[FP32]
                  |
                  |
             Output[FP32]
*/

TEST_F(AddConvertToReorderTest, smoke_TestAddConvert_CPU) {
    BuildGraph(ngraph::element::bf16);
    Run();
    CheckElementOfTypeCount("Convert", 1);
    CheckElementOfTypeCount("Reorder", 0);
}

/* Test insertion of the Reorder layer if there is one.

    Parameter[FP32]     Constant[I8]
          \                 /
           \               /
            \       Reorder[I32] (Is inserted by the MKLDNNGraphOptimizer)
             \           /
             Gather[FP32]
                  |
                  |
             Output[FP32]
*/
TEST_F(AddConvertToReorderTest, smoke_TestAddReorder_CPU) {
    BuildGraph(ngraph::element::i8);
    Run();
    CheckElementOfTypeCount("Convert", 0);
    CheckElementOfTypeCount("Reorder", 1);
}
} // namespace
} // namespace LayerTestsDefinitions