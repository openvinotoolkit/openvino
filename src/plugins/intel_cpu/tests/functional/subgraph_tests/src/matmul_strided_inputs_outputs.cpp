// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_models/builders.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "test_utils/cpu_test_utils.hpp"

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
        ov::ParameterVector concatInputParams {std::make_shared<ov::op::v0::Parameter>(ngPrec, concatShapes[0]),
                                               std::make_shared<ov::op::v0::Parameter>(ngPrec, concatShapes[1])};
        const auto concatOutputNodes = convert2OutputVector(castOps2Nodes<ov::op::v0::Parameter>(concatInputParams));
        const auto concat = std::make_shared<ov::op::v0::Concat>(concatOutputNodes, 2);

        const auto matMul1 = std::make_shared<ov::op::v0::MatMul>(split->output(0), concat, false, false);

        ov::Shape matmulShape{1, 1, 16, 8};
        ov::ParameterVector matmulInputParams {std::make_shared<ov::op::v0::Parameter>(ngPrec, ov::Shape(matmulShape))};

        const auto matMul2 = std::make_shared<ov::op::v0::MatMul>(split->output(1), matmulInputParams[0], false, false);

        const auto concatMatMuls = std::make_shared<ov::op::v0::Concat>(ov::NodeVector{matMul1, matMul2}, 2 /* 3rd axis */);

        ov::ParameterVector inputParams = {splitInputParams[0], concatInputParams[0], concatInputParams[1], matmulInputParams[0]};
        function = makeNgraphFunction(ngPrec, inputParams, concatMatMuls, "MatmulStridedInputsOutputs");
    }

private:
    template <class opType>
    ov::NodeVector castOps2Nodes(const std::vector<std::shared_ptr<opType>>& ops) {
        ov::NodeVector nodes;
        for (const auto& op : ops) {
            nodes.push_back(std::dynamic_pointer_cast<ov::Node>(op));
        }
        return nodes;
    }

    ov::OutputVector convert2OutputVector(const std::vector<std::shared_ptr<ov::Node>>& nodes) {
        ov::OutputVector outs;
        std::for_each(nodes.begin(), nodes.end(), [&outs](const std::shared_ptr<ov::Node>& n) {
            for (const auto& out_p : n->outputs()) {
                outs.push_back(out_p);
            }
        });
        return outs;
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

namespace {

INSTANTIATE_TEST_SUITE_P(smoke_Check,
                         MatmulStridedInputsOutputsTest,
                         ::testing::Values(ov::element::f32, ov::element::bf16),
                         MatmulStridedInputsOutputsTest::getTestCaseName);

}  // namespace

}  // namespace test
}  // namespace ov
