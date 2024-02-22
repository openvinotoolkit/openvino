// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

class AddConvertToReorderTest : virtual public SubgraphBaseStaticTest {
public:
    void BuildGraph(const ov::element::Type& secondInpType) {
        secondConstantType = secondInpType;
        int axis = 2;
        std::vector<int> indices = {0, 3, 2, 1};
        std::vector<size_t> indicesShape = {2, 2};
        std::vector<size_t> inputShape = {10, 20, 30, 40};

        ov::element::Type netPrecision = inType = outType = ov::element::f32;
        targetDevice = ov::test::utils::DEVICE_CPU;

        ASSERT_EQ(ov::shape_size(indicesShape), indices.size())
            << "Indices vector size and provided indices shape doesn't fit each other";
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(netPrecision, ov::Shape(inputShape))};
        auto indicesNode = ov::op::v0::Constant::create(secondConstantType, ov::Shape(indicesShape), indices);
        auto axisNode = ov::op::v0::Constant::create(ov::element::i64, ov::Shape({}), {axis});
        auto gather = std::make_shared<ov::op::v1::Gather>(params[0], indicesNode, axisNode);
        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(gather)};
        function = std::make_shared<ov::Model>(results, params, "gather");
    }
    std::vector<ov::Tensor> calculate_refs() override {
        // Convert the second input constant precision to i64 to run the reference function
        if (ov::element::i8 == secondConstantType) {
            convert_precisions.insert({ov::element::i8, ov::element::i64});
        } else if (ov::element::bf16 == secondConstantType) {
            convert_precisions.insert({ov::element::bf16, ov::element::i64});
        }
        return SubgraphBaseTest::calculate_refs();
    }

private:
    ov::element::Type secondConstantType;
};

namespace {

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
    BuildGraph(ov::element::i8);
    run();
    CheckNumberOfNodesWithType(compiledModel, "Convert", 0);
    CheckNumberOfNodesWithType(compiledModel, "Reorder", 1);
}
}  // namespace
}  // namespace test
}  // namespace ov
