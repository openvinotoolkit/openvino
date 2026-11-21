// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/opsets/opset10_decl.hpp"
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

class ConvertBoolMathFractionalTest : public SubgraphBaseStaticTest {
public:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;

        ov::ParameterVector inputParams{std::make_shared<ov::opset10::Parameter>(ov::element::f32, ov::Shape{6})};

        auto convertBool = std::make_shared<ov::opset10::Convert>(inputParams.front(), ov::element::boolean);
        auto convertF32 = std::make_shared<ov::opset10::Convert>(convertBool, ov::element::f32);

        ov::ResultVector results{std::make_shared<ov::opset10::Result>(convertF32)};
        function = std::make_shared<ov::Model>(results, inputParams, "ConvertBoolMathFractional");
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        auto tensor = ov::Tensor{funcInputs[0].get_element_type(), targetInputStaticShapes[0]};
        auto* data = tensor.data<float>();
        data[0] = 0.0f;
        data[1] = 0.25f;
        data[2] = 1.0f;
        data[3] = -0.5f;
        data[4] = -5.0f;
        data[5] = 0.001f;
        inputs.insert({funcInputs[0].get_node_shared_ptr(), tensor});
    }
};

TEST_F(ConvertBoolMathFractionalTest, smoke_CompareWithRefs) {
    run();
}

class ConvertBoolMathF16Test : public SubgraphBaseStaticTest {
public:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;

        ov::ParameterVector inputParams{std::make_shared<ov::opset10::Parameter>(ov::element::f16, ov::Shape{6})};

        auto convertBool = std::make_shared<ov::opset10::Convert>(inputParams.front(), ov::element::boolean);
        auto convertF16 = std::make_shared<ov::opset10::Convert>(convertBool, ov::element::f16);

        ov::ResultVector results{std::make_shared<ov::opset10::Result>(convertF16)};
        function = std::make_shared<ov::Model>(results, inputParams, "ConvertBoolMathF16");
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        auto tensor = ov::Tensor{funcInputs[0].get_element_type(), targetInputStaticShapes[0]};
        auto* data = tensor.data<ov::float16>();
        data[0] = ov::float16(0.0f);
        data[1] = ov::float16(0.25f);
        data[2] = ov::float16(1.0f);
        data[3] = ov::float16(-0.5f);
        data[4] = ov::float16(-5.0f);
        data[5] = ov::float16(0.001f);
        inputs.insert({funcInputs[0].get_node_shared_ptr(), tensor});
    }
};

TEST_F(ConvertBoolMathF16Test, smoke_CompareWithRefs) {
    run();
}

}  // namespace test
}  // namespace ov
