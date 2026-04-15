// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Test for GitHub issue #33233: Incorrect Floor result (0.0 instead of 1.0)
// in float16 inference when Cos output approaches 1.0.
//
// When a Cos node feeds into Floor with f16 intermediate precision,
// the f16 rounding of cos(6.27734375) ≈ 0.99998 must produce 1.0 in f16,
// so that floor(1.0) = 1.0. Without proper f16 I/O support in the Math node,
// the graph may compute entirely in f32, giving floor(0.99998) = 0.0.

#include <cmath>
#include <vector>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/floor.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

//   ┌──────────┐
//   │ Param f16│
//   └────┬─────┘
//        │ f16
//   ┌────┴─────┐
//   │   Cos    │
//   └────┬─────┘
//        │ f16
//   ┌────┴─────┐
//   │  Floor   │
//   └────┬─────┘
//        │ f16
//   ┌────┴─────┐
//   │  Result  │
//   └──────────┘

class CosFloorF16Test : public SubgraphBaseStaticTest {
public:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;

        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1});
        auto cos = std::make_shared<ov::op::v0::Cos>(param);
        auto floor = std::make_shared<ov::op::v0::Floor>(cos);

        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(floor)};
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{param}, "CosFloorF16");
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        // 6.27734375 is close to 2*pi, so cos(6.27734375) ≈ 0.99998 in f32.
        // When rounded to f16, this becomes exactly 1.0,
        // so floor(1.0) should be 1.0, not 0.0.
        auto tensor = ov::Tensor(ov::element::f16, targetInputStaticShapes[0]);
        auto* data = tensor.data<ov::float16>();
        data[0] = ov::float16(6.27734375f);
        inputs.insert({function->get_parameters()[0], tensor});
    }
};

TEST_F(CosFloorF16Test, CompareWithRefs) {
    // Compute expected output: cos(6.27734375) in f32, then round to f16, then floor.
    float input_f32 = 6.27734375f;
    float cos_f32 = std::cos(input_f32);
    ov::float16 cos_f16 = ov::float16(cos_f32);
    float cos_f16_as_f32 = static_cast<float>(cos_f16);
    float expected_f32 = std::floor(cos_f16_as_f32);
    ov::float16 expected_f16 = ov::float16(expected_f32);

    // Sanity check: the expected result should be 1.0
    ASSERT_FLOAT_EQ(static_cast<float>(expected_f16), 1.0f)
        << "Expected floor(round_to_f16(cos(6.27734375))) = 1.0";

    // Compile and infer on CPU directly (don't compare against Template reference,
    // which also computes entirely in f32 and gives the wrong answer 0.0).
    auto compiled = core->compile_model(function, targetDevice);
    auto infer = compiled.create_infer_request();

    auto input_tensor = ov::Tensor(ov::element::f16, ov::Shape{1});
    input_tensor.data<ov::float16>()[0] = ov::float16(6.27734375f);
    infer.set_input_tensor(input_tensor);
    infer.infer();

    auto output_tensor = infer.get_output_tensor();
    ASSERT_EQ(output_tensor.get_element_type(), ov::element::f16);
    auto* actual_data = output_tensor.data<ov::float16>();
    EXPECT_FLOAT_EQ(static_cast<float>(actual_data[0]), 1.0f)
        << "Floor(Cos(6.27734375)) in f16 should be 1.0, not " << static_cast<float>(actual_data[0]);
}

}  // namespace test
}  // namespace ov
