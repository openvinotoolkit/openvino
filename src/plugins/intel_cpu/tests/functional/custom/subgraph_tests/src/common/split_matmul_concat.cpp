// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils/cpu_test_utils.hpp"
#include "common_test_utils/node_builders/constant.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

/*
            ---------------
            |    Input    |
            ---------------
                   |
            ---------------
            |VariadicSplit|
            ---------------
              |         |
          ---------     |
          |MatMul |     |
          ---------     |
              |         |
            ---------------
            |   Concat    |
            ---------------
            |
            ---------------
            |   Output    |
            ---------------
*/

using SplitMatMulConcatParams = std::tuple<
    std::vector<InputShape>,            // input shapes
    std::pair<bool, bool>               // transposeA, transposeB
>;

class SplitMatMulConcatTest : public testing::WithParamInterface<SplitMatMulConcatParams>,
                                    virtual public SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<SplitMatMulConcatParams> obj) {
        std::vector<InputShape> inputShapes;
        std::pair<bool, bool> transpose;

        std::tie(inputShapes, transpose) = obj.param;

        std::ostringstream result;
        for (const auto& shape : inputShapes) {
            result << ov::test::utils::partialShape2str({shape.first}) << "_";
        }
        result << "TS=";
        for (const auto& shape : inputShapes) {
            result << "(";
            if (!shape.second.empty()) {
                auto itr = shape.second.begin();
                do {
                    result << ov::test::utils::vec2str(*itr);
                } while (++itr != shape.second.end() && result << "_");
            }
            result << ")_";
        }
        result << "transpose_a=" << transpose.first << "_";
        result << "transpose_b=" << transpose.second << "_";

        return result.str();
    }

protected:
    template<typename T>
    void transposeShape(T& shape) {
        OPENVINO_ASSERT(shape.size() > 1);
        std::swap(*(shape.end() - 1), *(shape.end() - 2));
    }

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;

        std::vector<InputShape> inputShapes;
        std::pair<bool, bool> transpose;

        std::tie(inputShapes, transpose) = this->GetParam();

        init_input_shapes(inputShapes);

        bool transpA = transpose.first;
        bool transpB = transpose.second;

        if (transpA) {
            transposeShape(inputDynamicShapes[0]);
            for (auto& shapes : targetStaticShapes) {
                transposeShape(shapes[0]);
            }
        }
        if (transpB) {
            transposeShape(inputDynamicShapes[1]);
            for (auto& shapes : targetStaticShapes) {
                transposeShape(shapes[1]);
            }
        }

        const auto& inShapeA = inputDynamicShapes[0];
        const auto& inShapeB = inputDynamicShapes[1];

        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ElementType::f32, inShapeA)};
        std::shared_ptr<Node> inputB = ov::test::utils::make_constant(ElementType::f32, inShapeB.get_shape());

        auto split_axis_op = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, std::vector<int64_t>{0});
        auto num_split = std::make_shared<ov::op::v0::Constant>(ov::element::u64, ov::Shape{2}, std::vector<size_t>{1, 1});
        auto split = std::make_shared<ov::op::v1::VariadicSplit>(params[0], split_axis_op, num_split);

        auto matMul = std::make_shared<ov::op::v0::MatMul>(split->output(0), inputB, transpA, transpB);

        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{matMul, split->output(1)}, 0);

        function = CPUTestsBase::makeNgraphFunction(ElementType::f32, params, concat, "FullyConnected");
    }
};

TEST_P(SplitMatMulConcatTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
}

namespace {

const std::vector<std::pair<bool, bool>> transposeParams = {
    {false, true},
};

const std::vector<std::vector<InputShape>> inputShapes2D = {
    static_shapes_to_test_representation({{2, 3}, {3, 3}}),
};

const auto testParams2D_FP32_smoke = ::testing::Combine(
    ::testing::ValuesIn(inputShapes2D),
    ::testing::ValuesIn(transposeParams));

INSTANTIATE_TEST_SUITE_P(smoke_FC_2D_FP32, SplitMatMulConcatTest, testParams2D_FP32_smoke,
                        SplitMatMulConcatTest::getTestCaseName);

} // namespace

}  // namespace test
}  // namespace ov
