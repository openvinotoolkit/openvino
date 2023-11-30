// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils/cpu_test_utils.hpp"
#include "ov_models/builders.hpp"

using namespace ngraph;
using namespace InferenceEngine;
using namespace CPUTestUtils;

namespace SubgraphTestsDefinitions {

using MatmulStridedInputsOutputsTestParams = Precision;

class MatmulStridedInputsOutputsTest : public testing::WithParamInterface<MatmulStridedInputsOutputsTestParams>,
                                       public CPUTestsBase,
                                       virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<MatmulStridedInputsOutputsTestParams> obj) {
        Precision netPrecision;
        netPrecision = obj.param;

        std::ostringstream result;
        result << "netPRC=" << netPrecision.name() << "_";

        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;
        Precision netPrecision;
        netPrecision = this->GetParam();
        const auto ngPrec = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

        SizeVector splitShape{1, 2, 1, 16};
        ov::ParameterVector splitInputParams {std::make_shared<ov::op::v0::Parameter>(ngPrec, ov::Shape(splitShape))};
        auto split_axis_op = std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::i64, ov::Shape{}, std::vector<int64_t>{1});
        auto split = std::make_shared<ov::op::v1::Split>(splitInputParams[0], split_axis_op, 2);

        std::vector<ov::Shape> concatShapes{{1, 1, 8, 8}, {1, 1, 8, 8}};
        ov::ParameterVector concatInputParams {std::make_shared<ov::op::v0::Parameter>(ngPrec, concatShapes[0]),
                                               std::make_shared<ov::op::v0::Parameter>(ngPrec, concatShapes[1])};
        const auto concatOutputNodes = helpers::convert2OutputVector(helpers::castOps2Nodes<op::Parameter>(concatInputParams));
        const auto concat = std::make_shared<ov::op::v0::Concat>(concatOutputNodes, 2);

        const auto matMul1 = std::make_shared<ov::op::v0::MatMul>(split->output(0), concat, false, false);

        SizeVector matmulShape{1, 1, 16, 8};
        ov::ParameterVector matmulInputParams {std::make_shared<ov::op::v0::Parameter>(ngPrec, ov::Shape(matmulShape))};

        const auto matMul2 = std::make_shared<ov::op::v0::MatMul>(split->output(1), matmulInputParams[0], false, false);

        const auto concatMatMuls = std::make_shared<ov::op::v0::Concat>(ov::NodeVector{matMul1, matMul2}, 2 /* 3rd axis */);

        ngraph::ParameterVector inputParams = {splitInputParams[0], concatInputParams[0], concatInputParams[1], matmulInputParams[0]};
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
    Run();
}

TEST_P(MatmulStridedInputsOutputsTest, CompareWithRefs_FP16) {
    if (!(ov::with_cpu_x86_avx512_core_fp16() || ov::with_cpu_x86_avx512_core_amx_fp16())) {
        GTEST_SKIP() << "Skipping test, platform don't support precision f16";
    }
    configuration.insert({ov::hint::inference_precision.name(), ov::element::f16.to_string()});

    Run();
}


namespace {

INSTANTIATE_TEST_SUITE_P(smoke_Check, MatmulStridedInputsOutputsTest,
                         ::testing::Values(Precision::FP32,
                                           Precision::BF16),
                         MatmulStridedInputsOutputsTest::getTestCaseName);

} // namespace

} // namespace SubgraphTestsDefinitions
