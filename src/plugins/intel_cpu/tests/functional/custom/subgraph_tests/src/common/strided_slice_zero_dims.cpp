// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "common_test_utils/node_builders/constant.hpp"

namespace ov {
namespace test {

/*
    param1 [56]   param2 [-1, -1, 768] (dynamic shape)
        \            |
         \           |
          \       shapeOf [4] (variable)
           \         |
            \        |
             \     Gather (get dynamic element) [1] (static value)
              \      |
               \     |   OtherConstants
                \    |      /
                 StridedSlice [47] (Static output shape)
                     |
                     |
                   Result
*/

class StridedSliceZeroDimsTest : public SubgraphBaseTest {
public:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;
        InputShape inpShape0 = {{}, {{56}}};
        InputShape inpShape1 = {{-1, -1, 768}, {{1, 544, 768}}};
        init_input_shapes({inpShape0, inpShape1});
        ov::ParameterVector inputParams;
        for (auto&& shape : inputDynamicShapes) {
            inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape));
        }
        auto end = std::make_shared<ov::op::v0::Constant>(element::i64, ov::Shape{1}, std::vector<int64_t>{2147483647});
        auto stride  = std::make_shared<ov::op::v0::Constant>(element::i64, ov::Shape{1}, std::vector<int64_t>{1});
        auto indices = std::make_shared<ov::op::v0::Constant>(element::i64, ov::Shape{1}, std::vector<int64_t>{1});
        auto axes = std::make_shared<ov::op::v0::Constant>(element::i64, ov::Shape{1}, std::vector<int64_t>{0});
        auto shapeOf = std::make_shared<ov::op::v3::ShapeOf>(inputParams[1]);
        auto gather = std::make_shared<ov::op::v8::Gather>(shapeOf, indices, axes);
        auto strided_slice = std::make_shared<ov::op::v1::StridedSlice>(inputParams.front(),
                                                                        gather,
                                                                        end,
                                                                        stride,
                                                                        std::vector<int64_t>{0},
                                                                        std::vector<int64_t>{0},
                                                                        std::vector<int64_t>{},
                                                                        std::vector<int64_t>{},
                                                                        std::vector<int64_t>{});
        NodeVector results{strided_slice};
        function = std::make_shared<ov::Model>(results, inputParams, "StridedSliceStaticShape");
    }
};

TEST_F(StridedSliceZeroDimsTest, smoke_CompareWithRefs) {
    run();
}

}  // namespace test
}  // namespace ov