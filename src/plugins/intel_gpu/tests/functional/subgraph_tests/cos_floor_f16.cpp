// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Test for #184635: Incorrect Floor result (0.0 instead of 1.0)
// in float16 inference when Cos output approaches 1.0.
// This guards the GPU ConvertPrecision pipeline that preserves f16 rounding
// at Math op boundaries during f16 inference.

#include <cmath>

#include "openvino/core/type/float16.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/floor.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/tensor.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace {

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

class CosFloorF16GPUTest : public ov::test::SubgraphBaseStaticTest {
public:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;

        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1});
        auto cos = std::make_shared<ov::op::v0::Cos>(param);
        auto floor = std::make_shared<ov::op::v0::Floor>(cos);

        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(floor)};
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{param}, "CosFloorF16");
    }
};

TEST_F(CosFloorF16GPUTest, CompareWithRefs) {
    // Compute expected output: cos(6.27734375) in f32, then round to f16, then floor.
    // 6.27734375 is close to 2*pi, so cos(6.27734375) ~= 0.99998 in f32.
    // When rounded to f16 this becomes exactly 1.0, so floor(1.0) should be 1.0, not 0.0.
    const float input_f32 = 6.27734375f;
    const float cos_f32 = std::cos(input_f32);
    const ov::float16 cos_f16 = ov::float16(cos_f32);
    const float expected_f32 = std::floor(static_cast<float>(cos_f16));
    ASSERT_FLOAT_EQ(expected_f32, 1.0f) << "Expected floor(round_to_f16(cos(6.27734375))) = 1.0";

    // Compile and infer on GPU directly (the Template reference computes entirely in f32
    // and would give the wrong answer 0.0). Pin f16 inference precision so the graph keeps
    // f16 semantics regardless of plugin defaults.
    auto compiled = core->compile_model(function, targetDevice, ov::hint::inference_precision(ov::element::f16));
    auto infer = compiled.create_infer_request();

    auto input_tensor = ov::Tensor(ov::element::f16, ov::Shape{1});
    input_tensor.data<ov::float16>()[0] = ov::float16(input_f32);
    infer.set_input_tensor(input_tensor);
    infer.infer();

    auto output_tensor = infer.get_output_tensor();
    ASSERT_EQ(output_tensor.get_element_type(), ov::element::f16);
    const auto* actual_data = output_tensor.data<ov::float16>();
    EXPECT_FLOAT_EQ(static_cast<float>(actual_data[0]), 1.0f)
        << "Floor(Cos(6.27734375)) in f16 should be 1.0, not " << static_cast<float>(actual_data[0]);
}

}  // namespace
