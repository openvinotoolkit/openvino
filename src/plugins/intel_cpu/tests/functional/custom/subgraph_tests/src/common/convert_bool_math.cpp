// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"

namespace ov {
namespace test {
//   ┌────────┐
//   │ Param  │
//   └───┬────┘
//       │ f32
//       │
//   ┌───┴────┐
//   │Convert │
//   └───┬────┘
//       │ bool
//       │
//   ┌───┴────┐
//   │Reshape │
//   └───┬────┘
//       │ bool
//       │
//   ┌───┴────┐         ┌────────┐
//   │Convert │         │ Param  │
//   └───┬────┘         └───┬────┘
//       │ f32              │ f32
//       │                  │
//       │     ┌────────┐   │
//       └─────┤ Add    ├───┘
//             └───┬────┘
//                 │ f32
//                 │
//             ┌───┴────┐
//             │Reshape │
//             └────────┘

class ConvertBoolMathTest : public SubgraphBaseStaticTest {
public:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;

        ov::ParameterVector inputParams{std::make_shared<ov::opset10::Parameter>(ov::element::f32, ov::Shape{24, 7}),
                                        std::make_shared<ov::opset10::Parameter>(ov::element::f32, ov::Shape{3, 8, 7})};

        auto inputConvert = std::make_shared<ov::opset10::Convert>(inputParams.front(), ov::element::boolean);

        auto reshapeConst = ov::opset10::Constant::create<int32_t>(ov::element::i32, ov::Shape{3}, {3, 8, 7});
        auto reshape = std::make_shared<ov::opset10::Reshape>(inputConvert, reshapeConst, false);

        auto secondConvert = std::make_shared<ov::opset10::Convert>(reshape, ov::element::f32);
        auto add = std::make_shared<ov::opset10::Add>(secondConvert, inputParams.back());

        ov::ResultVector results{std::make_shared<ov::opset10::Result>(add)};
        function = std::make_shared<ov::Model>(results, inputParams, "ConvertBoolMath");
    }
};

TEST_F(ConvertBoolMathTest, smoke_CompareWithRefs) {
    run();
}

} // namespace test
} // namespace ov