// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/cos_floor_f16.hpp"

#include <cmath>

#include "openvino/core/type/float16.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/floor.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/properties.hpp"

namespace ov {
namespace test {

void CosFloorF16TestBase::SetUp() {
    // Pin f16 inference precision so the graph keeps f16 semantics regardless of plugin defaults.
    configuration[ov::hint::inference_precision.name()] = ov::element::f16;

    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1});
    auto cos = std::make_shared<ov::op::v0::Cos>(param);
    auto floor = std::make_shared<ov::op::v0::Floor>(cos);

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(floor)};
    function = std::make_shared<ov::Model>(results, ov::ParameterVector{param}, "CosFloorF16");
}

void CosFloorF16TestBase::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();
    auto tensor = ov::Tensor(ov::element::f16, targetInputStaticShapes[0]);
    tensor.data<ov::float16>()[0] = ov::float16(input_value);
    inputs.insert({function->get_parameters()[0], tensor});
}

void CosFloorF16TestBase::check_floor_result() {
    // Compute expected output: cos(input_value) in f32, then round to f16, then floor.
    const float cos_f32 = std::cos(input_value);
    const ov::float16 cos_f16 = ov::float16(cos_f32);
    const float expected_f32 = std::floor(static_cast<float>(cos_f16));
    ASSERT_FLOAT_EQ(expected_f32, 1.0f) << "Expected floor(round_to_f16(cos(" << input_value << "))) = 1.0";

    auto compiled = core->compile_model(function, targetDevice, configuration);
    auto infer_request = compiled.create_infer_request();

    generate_inputs({ov::Shape{1}});
    for (const auto& [parameter, tensor] : inputs) {
        infer_request.set_tensor(parameter, tensor);
    }
    infer_request.infer();

    auto output_tensor = infer_request.get_output_tensor();
    ASSERT_EQ(output_tensor.get_element_type(), ov::element::f16);
    const auto* actual_data = output_tensor.data<ov::float16>();
    EXPECT_FLOAT_EQ(static_cast<float>(actual_data[0]), 1.0f)
        << "Floor(Cos(" << input_value << ")) in f16 should be 1.0, not " << static_cast<float>(actual_data[0]);
}

}  // namespace test
}  // namespace ov
