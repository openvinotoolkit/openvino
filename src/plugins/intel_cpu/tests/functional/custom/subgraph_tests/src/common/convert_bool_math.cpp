// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <array>
#include <cstdint>
#include <limits>
#include <vector>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/opsets/opset10_decl.hpp"
#include "openvino/runtime/core.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

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

TEST(ConvertBoolMath, smoke_SignedMinimumIsTrue) {
    std::array<int32_t, 5> values{std::numeric_limits<int32_t>::min(), -1, 0, 1, std::numeric_limits<int32_t>::max()};
    const ov::Shape shape{values.size()};
    auto input = std::make_shared<ov::opset10::Parameter>(ov::element::i32, shape);
    auto as_bool = std::make_shared<ov::opset10::Convert>(input, ov::element::boolean);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{as_bool}, ov::ParameterVector{input});
    auto request = ov::Core().compile_model(model, ov::test::utils::DEVICE_CPU).create_infer_request();
    request.set_input_tensor(ov::Tensor(ov::element::i32, shape, values.data()));
    request.infer();
    const auto output = request.get_output_tensor();
    const auto* data = output.data<const char>();
    EXPECT_EQ(std::vector<char>(data, data + output.get_size()), (std::vector<char>{1, 1, 0, 1, 1}));
}

} // namespace test
} // namespace ov
