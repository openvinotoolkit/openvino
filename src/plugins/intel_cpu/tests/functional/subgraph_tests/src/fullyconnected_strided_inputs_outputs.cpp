// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/partial_shape.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include "ov_models/builders.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"

using namespace ngraph;
using namespace InferenceEngine;
using namespace CPUTestUtils;

namespace SubgraphTestsDefinitions {

using FullyConnectedStridedInputsOutputsTestParams = std::tuple<Precision,
                                                                size_t>; // rank (2D or 3D)

class FullyConnectedStridedInputsOutputsTest : public testing::WithParamInterface<FullyConnectedStridedInputsOutputsTestParams>,
                                       public CPUTestsBase,
                                       virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<FullyConnectedStridedInputsOutputsTestParams> obj) {
        Precision netPrecision;
        size_t rank;
        std::tie(netPrecision, rank) = obj.param;

        std::ostringstream result;
        result << "netPRC=" << netPrecision.name() << "_";
        result << "rank=" << rank;

        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;
        Precision netPrecision;
        size_t rank;
        std::tie(netPrecision, rank) = this->GetParam();
        const auto ngPrec = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

        auto bcastTo3D = [](SizeVector& shape) {
            shape.insert(shape.begin(), 1);
        };

        SizeVector splitShape{2, 16};
        if (rank == 3) bcastTo3D(splitShape);

        ov::ParameterVector params {std::make_shared<ov::op::v0::Parameter>(ngPrec, ov::Shape(splitShape))};

        const auto splitAxis = rank == 3 ? 1 : 0;
        auto split_axis_op = std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::i64, ov::Shape{}, std::vector<int64_t>{splitAxis});
        auto split = std::make_shared<ov::op::v1::Split>(params[0], split_axis_op, 2);

        SizeVector fcWeightsShape{16, 8};
        if (rank == 3) bcastTo3D(fcWeightsShape);

        auto tensor = ov::test::utils::create_and_fill_tensor(ngPrec, fcWeightsShape);
        auto fc1secondInput = std::make_shared<ov::op::v0::Constant>(tensor);
        const auto fc1 = std::make_shared<ov::op::v0::MatMul>(split->output(0), fc1secondInput, false, false);

        auto tensorB = ov::test::utils::create_and_fill_tensor(ngPrec, fcWeightsShape);
        auto fc2secondInputB = std::make_shared<ov::op::v0::Constant>(tensorB);
        const auto fc2 = std::make_shared<ov::op::v0::MatMul>(split->output(1), fc2secondInputB, false, false);

        const auto fcConcatAxis = rank == 3 ? 1 : 0;
        const auto concatMatMuls = std::make_shared<ov::op::v0::Concat>(ov::NodeVector{fc1, fc2}, fcConcatAxis);

        function = makeNgraphFunction(ngPrec, params, concatMatMuls, "FullyConnectedStridedInputsOutputs");
    }
};

/* Network with two FullyConnected (FC) nodes and multiple inplace nodes
 * Test that MatMul node works correctly with strided inputs / outputs

                  Input
                    |
                    |
                    |
                    |
       Input      Split      Input
          \       /   \       /
           \     /     \     /
            \   /       \   /
             \ /         \ /
             FC          FC
               \         /
                \       /
                 \     /
                  \   /
                 Concat
                    |
                    |
                 Output
*/

TEST_P(FullyConnectedStridedInputsOutputsTest, CompareWithRefs) {
    Run();
}

namespace {

INSTANTIATE_TEST_SUITE_P(smoke_Check, FullyConnectedStridedInputsOutputsTest,
                         ::testing::Combine(::testing::Values(Precision::FP32, Precision::BF16),
                                            ::testing::Values(2, 3)),
                         FullyConnectedStridedInputsOutputsTest::getTestCaseName);

} // namespace

} // namespace SubgraphTestsDefinitions
