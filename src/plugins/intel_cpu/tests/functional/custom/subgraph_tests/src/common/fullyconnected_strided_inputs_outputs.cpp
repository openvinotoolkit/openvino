// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/core/partial_shape.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

using FullyConnectedStridedInputsOutputsTestParams = std::tuple<ov::element::Type,
                                                                size_t>;  // rank (2D or 3D)

class FullyConnectedStridedInputsOutputsTest
    : public testing::WithParamInterface<FullyConnectedStridedInputsOutputsTestParams>,
      public CPUTestsBase,
      virtual public SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<FullyConnectedStridedInputsOutputsTestParams> obj) {
        ov::element::Type netPrecision;
        size_t rank;
        std::tie(netPrecision, rank) = obj.param;

        std::ostringstream result;
        result << "netPRC=" << netPrecision.get_type_name() << "_";
        result << "rank=" << rank;

        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;
        ov::element::Type netPrecision;
        size_t rank;
        std::tie(netPrecision, rank) = this->GetParam();

        auto bcastTo3D = [](ov::Shape& shape) {
            shape.insert(shape.begin(), 1);
        };

        ov::Shape splitShape{2, 16};
        if (rank == 3)
            bcastTo3D(splitShape);

        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(netPrecision, ov::Shape(splitShape))};

        const auto splitAxis = rank == 3 ? 1 : 0;
        auto split_axis_op = std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::i64,
                                                                    ov::Shape{},
                                                                    std::vector<int64_t>{splitAxis});
        auto split = std::make_shared<ov::op::v1::Split>(params[0], split_axis_op, 2);

        ov::Shape fcWeightsShape{16, 8};
        if (rank == 3)
            bcastTo3D(fcWeightsShape);

        auto tensor = ov::test::utils::create_and_fill_tensor(netPrecision, fcWeightsShape);
        auto fc1secondInput = std::make_shared<ov::op::v0::Constant>(tensor);
        const auto fc1 = std::make_shared<ov::op::v0::MatMul>(split->output(0), fc1secondInput, false, false);

        auto tensorB = ov::test::utils::create_and_fill_tensor(netPrecision, fcWeightsShape);
        auto fc2secondInputB = std::make_shared<ov::op::v0::Constant>(tensorB);
        const auto fc2 = std::make_shared<ov::op::v0::MatMul>(split->output(1), fc2secondInputB, false, false);

        const auto fcConcatAxis = rank == 3 ? 1 : 0;
        const auto concatMatMuls = std::make_shared<ov::op::v0::Concat>(ov::NodeVector{fc1, fc2}, fcConcatAxis);

        function = makeNgraphFunction(netPrecision, params, concatMatMuls, "FullyConnectedStridedInputsOutputs");
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
    run();
}

using FullyConnectedStridedInputsOutputsTest_FP16 = FullyConnectedStridedInputsOutputsTest;
TEST_P(FullyConnectedStridedInputsOutputsTest_FP16, CompareWithRefs) {
    if (!(ov::with_cpu_x86_avx512_core_fp16())) {
        GTEST_SKIP() << "Skipping test, platform don't support precision f16";
    }
    configuration.insert({ov::hint::inference_precision.name(), ov::element::f16});

    run();
}

namespace {

INSTANTIATE_TEST_SUITE_P(smoke_Check,
                         FullyConnectedStridedInputsOutputsTest,
                         ::testing::Combine(::testing::Values(ov::element::f32, ov::element::bf16),
                                            ::testing::Values(2, 3)),
                         FullyConnectedStridedInputsOutputsTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Check,
                         FullyConnectedStridedInputsOutputsTest_FP16,
                         ::testing::Combine(::testing::Values(ov::element::f32),
                                            ::testing::Values(2, 3)),
                         FullyConnectedStridedInputsOutputsTest::getTestCaseName);

}  // namespace

}  // namespace test
}  // namespace ov
