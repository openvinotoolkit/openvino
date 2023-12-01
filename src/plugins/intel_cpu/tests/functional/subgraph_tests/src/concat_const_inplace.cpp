// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_models/builders.hpp"
#include "ov_models/utils/ov_helpers.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "test_utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
// Subgraph:
/*
 *          Parameter    Constant[FP32/BF16]
 *                  \    /
 *                   \  /
 *               Transpose[FP32/BF16]
 *  Constant[FP32] /
 *        \      X  No Reorder
 *         \    /
 *        Concat (inPlace)[FP32/BF16]
 *           |
 *      Convolution [FP32/BF16]
 *           |
 *        Result[FP32/BF16]
 */

class ConcatConstantInPlaceTest : public testing::WithParamInterface<ov::element::Type>,
                                  virtual public SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ov::element::Type> obj) {
        std::ostringstream result;
        result << "ConcatConstantInPlaceTest" << obj.param.get_type_name();
        return result.str();
    }

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;
        if (ov::element::bf16 == (inType = outType = this->GetParam()))
            configuration.insert({ov::hint::inference_precision.name(), ov::element::bf16});

        const ov::Shape inputShape = {1, 3, 3, 11};
        ov::ParameterVector inputParams{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputShape)};

        auto transposeOrder = ov::op::v0::Constant::create(ov::element::i32, {4}, {0, 3, 2, 1});
        auto transpose = std::make_shared<ov::op::v1::Transpose>(inputParams[0], transposeOrder);

        auto concatConstantInput = ov::op::v0::Constant::create(ov::element::f32, {1, 1, 3, 3}, {10.0f});
        auto concat = std::make_shared<ov::op::v0::Concat>(ov::NodeVector{concatConstantInput, transpose}, 1);

        // convolution
        std::vector<float> weightValuesFP32(12);
        ov::Shape convFilterShape = {1, 12, 1, 1};
        FuncTestUtils::fillInputsBySinValues(weightValuesFP32.data(), weightValuesFP32.size());
        auto weightsNode = std::make_shared<ov::op::v0::Constant>(ov::element::f32, convFilterShape, weightValuesFP32);
        std::shared_ptr<ov::Node> conv = std::make_shared<ov::op::v1::Convolution>(concat,
                                                                                   weightsNode,
                                                                                   ov::Strides({1, 1}),
                                                                                   ov::CoordinateDiff({0, 0}),
                                                                                   ov::CoordinateDiff({0, 0}),
                                                                                   ov::Strides({1, 1}),
                                                                                   ov::op::PadType::EXPLICIT);
        conv->set_friendly_name("CONV");

        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(conv)};
        function = std::make_shared<ov::Model>(results, inputParams, "ConcatConstantInPlace");
    }
};

namespace {
TEST_P(ConcatConstantInPlaceTest, smoke_ConcatConstantInPlaceTest_CPU) {
    run();
    if (this->GetParam() == ov::element::bf16)
        CheckNumberOfNodesWithType(compiledModel, "Reorder", 3);
    else
        CheckNumberOfNodesWithType(compiledModel, "Reorder", 2);
}

INSTANTIATE_TEST_SUITE_P(smoke_ConcatConstantInPlaceTest_CPU,
                         ConcatConstantInPlaceTest,
                         testing::Values(ov::element::f32, ov::element::bf16),
                         ConcatConstantInPlaceTest::getTestCaseName);

}  // namespace
}  // namespace test
}  // namespace ov
