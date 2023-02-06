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
    param1 [56]   param2 [1..5, 3, 4, 2] (dynamic shape)
        \            |
         \           |
          \       shapeOf [4] (variable)
           \         |
            \        |
             \     Gather (only last element) [1] (static value)
              \      |
               \     |   OtherConstants
                \    |      /
                 StridedSlice [47] (Static output shape)
                     |
                     |
                   Result
*/

class StridedSliceStaticShapeTest : public SubgraphBaseTest {
public:
    void SetUp() override {
        targetDevice = CommonTestUtils::DEVICE_CPU;
        InputShape inpShape0 = {{}, {{56}}};
        InputShape inpShape1 = {{{1, 5}, 3, 4, 2}, {{2, 3, 4, 2}}};
        init_input_shapes({inpShape0, inpShape1});
        auto inputParams = builder::makeDynamicParams(element::f32, inputDynamicShapes);
        auto end = builder::makeConstant(element::i64, {1}, std::vector<int64_t>{50});
        auto stride  = builder::makeConstant(element::i64, {1}, std::vector<int64_t>{1});
        auto indices = builder::makeConstant(element::i64, {1}, std::vector<int64_t>{3});
        auto axes = builder::makeConstant(element::i64, {1}, std::vector<int64_t>{0});
        auto shapeOf = std::make_shared<opset9::ShapeOf>(inputParams[1]);
        auto gather = std::make_shared<opset9::Gather>(shapeOf, indices, axes);
        auto strided_slice = builder::makeStridedSlice(inputParams.front(), gather, end, stride, element::f32, {0}, {0});
        NodeVector results{strided_slice};
        function = std::make_shared<Function>(results, inputParams, "StridedSliceStaticShape");
        ov::pass::Serialize serializer("init_graph.xml", "init_graph.bin");
        serializer.run_on_model(std::const_pointer_cast<ov::Model>(compiledModel.get_runtime_model()));
    }
};

TEST_F(StridedSliceStaticShapeTest, smoke_CompareWithRefs) {
    run();
}

} // namespace SubgraphTestsDefinitions