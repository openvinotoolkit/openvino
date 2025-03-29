// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

using MatmulStridedInputsOutputsTestParams = ov::element::Type;

class MatmulStridedInputsOutputsTest : public testing::WithParamInterface<MatmulStridedInputsOutputsTestParams>,
                                       public CPUTestsBase,
                                       virtual public SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<MatmulStridedInputsOutputsTestParams> obj) {
        ov::element::Type netPrecision;
        netPrecision = obj.param;

        std::ostringstream result;
        result << "netPRC=" << netPrecision.to_string() << "_";

        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = utils::DEVICE_CPU;
        const auto ngPrec = this->GetParam();

        ov::Shape splitShape{1, 2, 1, 16};
        ov::ParameterVector splitInputParams {std::make_shared<ov::op::v0::Parameter>(ngPrec, ov::Shape(splitShape))};
        auto split_axis_op = std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::i64, ov::Shape{}, std::vector<int64_t>{1});
        auto split = std::make_shared<ov::op::v1::Split>(splitInputParams[0], split_axis_op, 2);

        std::vector<ov::Shape> concatShapes{{1, 1, 8, 8}, {1, 1, 8, 8}};
        ov::ParameterVector concatInputParams{std::make_shared<ov::op::v0::Parameter>(ngPrec, concatShapes[0]),
                                              std::make_shared<ov::op::v0::Parameter>(ngPrec, concatShapes[1])};
        ov::OutputVector concatOutputNodes;
        for (auto&& node : concatInputParams) {
            for (auto&& param : node->outputs())
                concatOutputNodes.push_back(param);
        }

        const auto concat = std::make_shared<ov::op::v0::Concat>(concatOutputNodes, 2);

        const auto matMul1 = std::make_shared<ov::op::v0::MatMul>(split->output(0), concat, false, false);

        ov::Shape matmulShape{1, 1, 16, 8};
        ov::ParameterVector matmulInputParams {std::make_shared<ov::op::v0::Parameter>(ngPrec, ov::Shape(matmulShape))};

        const auto matMul2 = std::make_shared<ov::op::v0::MatMul>(split->output(1), matmulInputParams[0], false, false);

        const auto concatMatMuls = std::make_shared<ov::op::v0::Concat>(ov::NodeVector{matMul1, matMul2}, 2 /* 3rd axis */);

        ov::ParameterVector inputParams = {splitInputParams[0], concatInputParams[0], concatInputParams[1], matmulInputParams[0]};
        function = makeNgraphFunction(ngPrec, inputParams, concatMatMuls, "MatmulStridedInputsOutputs");
    }
};

/* Network with two MatMul nodes and multiple inplace nodes
 * Test that MatMul node works correctly with strided inputs / outputs

   Input    Input Input
     \       /      |
      \     /       |
       \   /        |
        \ /         |
       Concat     Split      Input
          \       /   \       /
           \     /     \     /
            \   /       \   /
             \ /         \ /
            MatMul     MatMul
               \         /
                \       /
                 \     /
                  \   /
                 Concat
                    |
                    |
                 Output
*/

TEST_P(MatmulStridedInputsOutputsTest, CompareWithRefs) {
    run();
}

using MatmulStridedInputsOutputsTest_FP16 = MatmulStridedInputsOutputsTest;
TEST_P(MatmulStridedInputsOutputsTest_FP16, CompareWithRefs) {
    if (!(ov::with_cpu_x86_avx512_core_fp16())) {
        GTEST_SKIP() << "Skipping test, platform don't support precision f16";
    }
    configuration.insert({ov::hint::inference_precision.name(), ov::element::f16});

    run();
}

namespace {

INSTANTIATE_TEST_SUITE_P(smoke_Check,
                         MatmulStridedInputsOutputsTest,
                         ::testing::Values(ov::element::f32, ov::element::bf16),
                         MatmulStridedInputsOutputsTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Check,
                         MatmulStridedInputsOutputsTest_FP16,
                         ::testing::Values(ov::element::f32),
                         MatmulStridedInputsOutputsTest::getTestCaseName);

}  // namespace

}  // namespace test
}  // namespace ov
