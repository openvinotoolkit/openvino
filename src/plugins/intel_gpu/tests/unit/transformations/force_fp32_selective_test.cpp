// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef GPU_DEBUG_CONFIG

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "intel_gpu/op/fully_connected.hpp"
#include "intel_gpu/op/fully_connected_compressed.hpp"
#include "intel_gpu/op/gemm.hpp"
#include "intel_gpu/op/indirect_sdpa.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/select.hpp"
#include "openvino/pass/manager.hpp"
#include "ov_ops/rms.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "plugin/transformations/force_fp32_selective.hpp"

namespace {

void force_selected(const std::shared_ptr<ov::Model>& model) {
    ov::pass::Manager manager;
    manager.register_pass<ov::intel_gpu::ForceFP32Selective>(std::vector<std::string>{},
                                                          std::vector<std::string>{"selected"});
    manager.run_passes(model);
    model->validate_nodes_and_infer_types();
}

// Check both explicit f16 and dynamic attributes; only the former should change.
class ForceFP32SelectiveRMS : public testing::TestWithParam<ov::element::Type> {};

TEST_P(ForceFP32SelectiveRMS, PreservesPrecisionAfterValidationAndClone) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 4});
    auto selected = std::make_shared<ov::op::internal::RMS>(input, 1e-5, GetParam());
    auto unselected = std::make_shared<ov::op::internal::RMS>(input, 1e-5, ov::element::f16);
    selected->set_friendly_name("selected");
    unselected->set_friendly_name("other");
    auto model = std::make_shared<ov::Model>(ov::OutputVector{selected, unselected}, ov::ParameterVector{input});

    force_selected(model);

    EXPECT_EQ(selected->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(unselected->get_output_element_type(0), ov::element::f16);
    EXPECT_EQ(input->get_element_type(), ov::element::f16);
    EXPECT_EQ(model->get_output_element_type(0), ov::element::f16);
    auto probe = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 4});
    auto clone = selected->clone_with_new_inputs({probe});
    EXPECT_EQ(clone->get_output_element_type(0),
              GetParam() == ov::element::dynamic ? ov::element::f16 : ov::element::f32);
    EXPECT_DOUBLE_EQ(selected->get_epsilon(), 1e-5);
}

INSTANTIATE_TEST_SUITE_P(PrecisionAttributes,
                         ForceFP32SelectiveRMS,
                         testing::Values(ov::element::f16, ov::element::dynamic));

TEST(ForceFP32Selective, ConvertDestinationSurvivesValidation) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4});
    auto selected = std::make_shared<ov::op::v0::Convert>(input, ov::element::f16);
    auto other = std::make_shared<ov::op::v0::Convert>(input, ov::element::f16);
    selected->set_friendly_name("selected");
    other->set_friendly_name("other");
    auto model = std::make_shared<ov::Model>(ov::OutputVector{selected, other}, ov::ParameterVector{input});

    force_selected(model);

    EXPECT_EQ(selected->get_destination_type(), ov::element::f32);
    EXPECT_EQ(selected->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(other->get_destination_type(), ov::element::f16);
    EXPECT_EQ(model->get_output_element_type(0), ov::element::f16);
}

TEST(ForceFP32Selective, FullyConnectedOutputAttributeSurvivesValidation) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 4});
    auto weights = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{2, 4}, {1});
    auto bias = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{2}, {0});
    auto selected = std::make_shared<ov::intel_gpu::op::FullyConnected>(input, weights, bias, ov::element::f16);
    auto other = std::make_shared<ov::intel_gpu::op::FullyConnected>(input, weights, bias, ov::element::f16);
    selected->set_friendly_name("selected");
    other->set_friendly_name("other");
    auto model = std::make_shared<ov::Model>(ov::OutputVector{selected, other}, ov::ParameterVector{input});

    force_selected(model);

    EXPECT_EQ(selected->get_output_type(), ov::element::f32);
    EXPECT_EQ(selected->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(other->get_output_type(), ov::element::f16);
    EXPECT_EQ(weights->get_element_type(), ov::element::f16);
    EXPECT_TRUE(selected->get_transpose_b());
}

TEST(ForceFP32Selective, CompressedFullyConnectedUsesInheritedAttributeVisitor) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 4});
    auto weights = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{2, 4}, {1});
    auto bias = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{2}, {0});
    auto scale = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{2, 1}, {1});
    auto selected = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input,
                                                                               weights,
                                                                               bias,
                                                                               scale,
                                                                               ov::element::f16);
    selected->set_friendly_name("selected");
    auto model = std::make_shared<ov::Model>(ov::OutputVector{selected}, ov::ParameterVector{input});

    force_selected(model);

    EXPECT_EQ(selected->get_output_type(), ov::element::f32);
    EXPECT_EQ(selected->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(weights->get_element_type(), ov::element::u8);
    EXPECT_EQ(scale->get_element_type(), ov::element::f16);
    auto clone = selected->clone_with_new_inputs(selected->input_values());
    EXPECT_TRUE(ov::is_type<ov::intel_gpu::op::FullyConnectedCompressed>(clone));
    EXPECT_EQ(clone->get_output_element_type(0), ov::element::f32);
}

TEST(ForceFP32Selective, GemmKeepsTransposeOrders) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 4});
    auto weights = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{2, 4});
    const std::vector<int64_t> order_a{0, 1};
    const std::vector<int64_t> order_b{1, 0};
    const std::vector<int64_t> order_c{0, 1};
    auto selected = std::make_shared<ov::intel_gpu::op::Gemm>(input,
                                                           weights,
                                                           order_a,
                                                           order_b,
                                                           order_c,
                                                           ov::element::f16);
    selected->set_friendly_name("selected");
    auto model = std::make_shared<ov::Model>(ov::OutputVector{selected}, ov::ParameterVector{input, weights});

    force_selected(model);

    EXPECT_EQ(selected->get_output_type(), ov::element::f32);
    EXPECT_EQ(selected->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(selected->get_input0_transpose_order(), order_a);
    EXPECT_EQ(selected->get_input1_transpose_order(), order_b);
    EXPECT_EQ(selected->get_output_transpose_order(), order_c);
}

TEST(ForceFP32Selective, IndirectSDPAKeepsDerivedAttributesAndIntegerInput) {
    auto query = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 2, 3, 4});
    auto key = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 2, 3, 4});
    auto value = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 2, 3, 4});
    auto beam = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{1, 3});
    const std::vector<int64_t> order{0, 1, 2, 3};
    auto selected = std::make_shared<ov::intel_gpu::op::IndirectSDPA>(ov::OutputVector{query, key, value},
                                                                   beam,
                                                                   true,
                                                                   0,
                                                                   order,
                                                                   order,
                                                                   order,
                                                                   order,
                                                                   ov::element::f16);
    selected->set_friendly_name("selected");
    auto model = std::make_shared<ov::Model>(ov::OutputVector{selected},
                                           ov::ParameterVector{query, key, value, beam});

    force_selected(model);

    EXPECT_EQ(selected->get_output_type(), ov::element::f32);
    EXPECT_EQ(selected->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(selected->get_indirect_axis(), 0);
    EXPECT_TRUE(selected->get_causal());
    EXPECT_EQ(selected->get_input0_transpose_order(), order);
    EXPECT_EQ(selected->get_input_element_type(3), ov::element::i32);
    EXPECT_EQ(selected->input_value(3).get_node_shared_ptr(), beam);
}

TEST(ForceFP32Selective, TypeRelaxedChangesOnlyFP16Overrides) {
    auto condition = std::make_shared<ov::op::v0::Parameter>(ov::element::boolean, ov::Shape{1, 4});
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 4});
    auto selected = std::make_shared<ov::op::TypeRelaxed<ov::op::v1::Select>>(
        ov::op::v1::Select(condition, input, input),
        ov::element::TypeVector{ov::element::boolean, ov::element::f16, ov::element::dynamic},
        ov::element::TypeVector{ov::element::f16});
    selected->set_friendly_name("selected");
    auto model = std::make_shared<ov::Model>(ov::OutputVector{selected}, ov::ParameterVector{condition, input});

    force_selected(model);

    EXPECT_EQ(selected->get_origin_input_type(0), ov::element::boolean);
    EXPECT_EQ(selected->get_origin_input_type(1), ov::element::f32);
    EXPECT_EQ(selected->get_origin_input_type(2), ov::element::dynamic);
    EXPECT_EQ(selected->get_overridden_output_type(0), ov::element::f32);
    EXPECT_EQ(selected->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(condition->get_element_type(), ov::element::boolean);
    EXPECT_EQ(model->get_output_element_type(0), ov::element::f16);
}

}  // namespace

#endif  // GPU_DEBUG_CONFIG
