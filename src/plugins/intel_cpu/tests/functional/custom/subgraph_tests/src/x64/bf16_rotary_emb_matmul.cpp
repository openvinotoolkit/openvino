// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/activation.hpp"
#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov::test {

// Here we test BF16 path for the LLM rotary embedding matmul operation which has one of the inputs converted from
// int64 to bf16. In the CPU plugin we have a special pattern matcher that prevents bf16 markup in this particular case.

//    ┌──────────────┐          ┌──────────────┐
//    │              │          │              │
//    │ Param 1 int64│          │ Param 2 float│
//    │              │          │              │
//    └──────┬───────┘          └───────┬──────┘
//           │                          │
//           │                          │
//    ┌──────┴───────┐                  │
//    │              │                  │
//    │ Convert float│                  │
//    │              │                  │
//    └──────┬───────┘                  │
//           │                          │
//           │                          │
//           │                          │
//           │       ┌──────────────┐   │
//           │       │              │   │
//           └───────┤ MatMul1 float├───┘
//                   │              │
//                   └──────┬───────┘
//                          │
//                          │
//                          │
//                   ┌──────┴────────┐
//                   │               │
//                   │    Cos float  │
//                   │               │
//                   └──────┬────────┘
//                          │
//                          │
//                   ┌──────┴───────┐
//                   │              │
//                   │ Result1 float│
//                   │              │
//                   └──────────────┘

class BF16RotaryEmbMatMul : public SubgraphBaseTest {
public:
    void SetUp() override {
        const std::vector<InputShape> input_shapes = {
            {{-1, 1, -1}, {{2, 1, 32}}},    // param 0
            {{-1, 32, -1}, {{2, 32, 32}}},  // param 1
        };

        init_input_shapes(input_shapes);

        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::i64, inputDynamicShapes[0]),
                                   std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes[1])};

        auto convert = std::make_shared<ov::op::v0::Convert>(params.front(), ov::element::f32);
        auto matmul = std::make_shared<ov::op::v0::MatMul>(convert, params[1]);
        auto cos = utils::make_activation(matmul, ov::element::f32, utils::ActivationTypes::Cos);
        function = std::make_shared<ov::Model>(ov::OutputVector{std::make_shared<ov::op::v0::Result>(cos)},
                                               params,
                                               "BF16RotaryEmbMatMul");
        targetDevice = utils::DEVICE_CPU;

        configuration.insert({ov::hint::inference_precision.name(), ov::element::bf16});
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];

            ov::Tensor tensor;

            if (funcInput.get_element_type() == ov::element::i64) {
                utils::InputGenerateData in_data;
                in_data.start_from = 128;
                in_data.range = 256;
                in_data.resolution = 1;
                tensor =
                    utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i], in_data);
            } else {
                utils::InputGenerateData in_data;
                in_data.start_from = 0;
                in_data.range = 15;
                in_data.resolution = 32;
                tensor =
                    utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i], in_data);
            }
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }
};

TEST_F(BF16RotaryEmbMatMul, smoke_BF16RotaryEmbMatMul_CPU) {
    if (!ov::with_cpu_x86_bfloat16()) {
        GTEST_SKIP();
    }
    run();
}

}  // namespace ov::test