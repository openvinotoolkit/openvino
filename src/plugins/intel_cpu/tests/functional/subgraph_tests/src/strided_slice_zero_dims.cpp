// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"

using namespace InferenceEngine;
using namespace ov::test;
using namespace ngraph;

namespace SubgraphTestsDefinitions {

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
        targetDevice = CommonTestUtils::DEVICE_CPU;
        InputShape inpShape0 = {{}, {{56}}};
        InputShape inpShape1 = {{-1, -1, 768}, {{1, 544, 768}}};
        init_input_shapes({inpShape0, inpShape1});
        auto inputParams = builder::makeDynamicParams(element::f32, inputDynamicShapes);
        auto end = builder::makeConstant(element::i64, {1}, std::vector<int64_t>{2147483647});
        auto stride  = builder::makeConstant(element::i64, {1}, std::vector<int64_t>{1});
        auto indices = builder::makeConstant(element::i64, {1}, std::vector<int64_t>{1});
        auto axes = builder::makeConstant(element::i64, {1}, std::vector<int64_t>{0});
        auto shapeOf = std::make_shared<opset9::ShapeOf>(inputParams[1]);
        auto gather = std::make_shared<opset9::Gather>(shapeOf, indices, axes);
        auto strided_slice = builder::makeStridedSlice(inputParams.front(), gather, end, stride, element::f32, {0}, {0});
        NodeVector results{strided_slice};
        function = std::make_shared<Function>(results, inputParams, "StridedSliceStaticShape");
    }
};

TEST_F(StridedSliceZeroDimsTest, smoke_CompareWithRefs) {
    run();
}

} // namespace SubgraphTestsDefinitions