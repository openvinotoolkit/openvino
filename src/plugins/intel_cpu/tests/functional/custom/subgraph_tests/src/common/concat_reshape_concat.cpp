// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "common_test_utils/node_builders/constant.hpp"

/*This test runs the following subgraph:

    param1      param2      param3      param4
       |          |          |           |
       |          |          |           |
    Softmax     Softmax    Softmax     Softmax
       |          |          |           |
       |          |          |           |
    Reshape     Reshape    Reshape     Reshape
       |          |          |           |
       |          |          |           |
       \          /          \          /
        \        /            \        /
         \      /              \      /
          Concat                Concat
               |                |
               |                |
              Reshape           Reshape
                    |           |
                    \          /
                     \        /
                      \      /
                       Concat
                          |
                        Softmax
                          
                        Result
  
  The main purpose of this test is checking the code path when all the nodes except Softmax use "in-place" memory mode.
  Softmax is used as a model of an arbitrary subgraph preceding the pattern.
*/

namespace ov {
namespace test {

using VectorShapes = std::vector<InputShape>;

class ConcatReshapeConcatSubgraphTest : public testing::WithParamInterface<VectorShapes>,
                                        virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<VectorShapes> obj) {
        VectorShapes& inputShapes = obj.param;

        std::ostringstream result;
        result << "IS=";
        for (const auto& shape : inputShapes) {
            result << ov::test::utils::partialShape2str({shape.first}) << "_";
        }
        result << "TS=";
        for (const auto& shape : inputShapes) {
            result << "(";
            if (!shape.second.empty()) {
                for (const auto& itr : shape.second) {
                    result << ov::test::utils::vec2str(itr);
                }
            }
            result << ")";
        }
        return result.str();
    }

    void SetUp() override {
        constexpr size_t number_of_params = 4ul;
        constexpr size_t softmax_axis = 1ul;
        constexpr int concat_axis = 0;
        targetDevice = ov::test::utils::DEVICE_CPU;
        auto netPrc = ov::element::f32;
        auto& InputShapes = this->GetParam();
        ASSERT_EQ(InputShapes.size(), number_of_params) << "Unexpected number of input shapes";
        init_input_shapes(InputShapes);
        ov::ParameterVector input_params;
        for (auto&& shape : inputDynamicShapes) {
            input_params.push_back(std::make_shared<ov::op::v0::Parameter>(netPrc, shape));
        }
        ov::NodeVector first_level_reshapes;

        for (size_t i = 0; i < number_of_params; ++i) {
            auto soft_max = std::make_shared<ov::op::v1::Softmax>(input_params[i], softmax_axis);
            auto reshape_param = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1}, std::vector<int>{0});
            auto reshape = std::make_shared<ov::op::v0::Unsqueeze>(soft_max, reshape_param);
            first_level_reshapes.push_back(reshape);
        }

        auto concat1 = std::make_shared<ov::op::v0::Concat>(ov::NodeVector{first_level_reshapes[0], first_level_reshapes[1]}, concat_axis);
        auto concat2 = std::make_shared<ov::op::v0::Concat>(ov::NodeVector{first_level_reshapes[2], first_level_reshapes[3]}, concat_axis);

        ov::NodeVector second_level_reshapes;
        ov::NodeVector first_level_concats = {concat1, concat2};

        for (size_t i = 0; i < number_of_params / 2; ++i) {
            auto reshape_param = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1}, std::vector<int>{0});
            auto reshape = std::make_shared<ov::op::v0::Unsqueeze>(first_level_concats[i], reshape_param);
            second_level_reshapes.push_back(reshape);
        }

        auto concat3 = std::make_shared<ov::op::v0::Concat>(second_level_reshapes, concat_axis);
        auto soft_max = std::make_shared<ov::op::v1::Softmax>(concat3, softmax_axis);

        ov::ResultVector results;
        for (size_t i = 0; i < soft_max->get_output_size(); i++)
            results.push_back(std::make_shared<ov::op::v0::Result>(soft_max->output(i)));

        function = std::make_shared<ov::Model>(results, input_params, "ConcatReshapeConcatPattern");
    }
};

TEST_P(ConcatReshapeConcatSubgraphTest, CompareWithRefs) {
    run();
}

namespace {

const std::vector<std::vector<InputShape>> inputShapes = {
    {
        // {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
        {{2, 64}, {{2, 64}}}, // input 0
        {{2, 64}, {{2, 64}}}, // input 1
        {{2, 64}, {{2, 64}}}, // input 2
        {{2, 64}, {{2, 64}}}  // input 3
    },
    {
        // {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
        {{2, -1}, {{2, 64}}}, // input 0
        {{2, -1}, {{2, 64}}}, // input 1
        {{2, -1}, {{2, 64}}}, // input 2
        {{2, -1}, {{2, 64}}}  // input 3
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_Concat_Reshape_Concat, ConcatReshapeConcatSubgraphTest,
                        ::testing::ValuesIn(inputShapes),
                        ConcatReshapeConcatSubgraphTest::getTestCaseName);
} // namespace
}  // namespace test
}  // namespace ov