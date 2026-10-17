// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef GPU_DEBUG_CONFIG

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "intel_gpu/op/fully_connected.hpp"
#include "intel_gpu/op/placeholder.hpp"
#include "openvino/core/any.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/runtime/tensor.hpp"
#include "ov_ops/rms.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "plugin/transformations/force_fp32_selective.hpp"

namespace force_fp32_selective_test {

class MixedOutputOp : public ov::op::Op {
public:
    OPENVINO_OP("MixedOutputOp", "test");

    explicit MixedOutputOp(const ov::Output<ov::Node>& input) : Op({input}) {
        validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        const auto shape = get_input_partial_shape(0);
        set_output_type(0, ov::element::i32, shape);
        set_output_type(1, get_input_element_type(0), shape);
        set_output_type(2, ov::element::f32, shape);
        set_output_type(3, ov::element::boolean, shape);
        set_output_type(4, ov::element::bf16, shape);
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return std::make_shared<MixedOutputOp>(inputs.at(0));
    }
};


}  // namespace force_fp32_selective_test

namespace {

using force_fp32_selective_test::MixedOutputOp;

void force_selected(const std::shared_ptr<ov::Model>& model) {
    ov::pass::Manager manager;
    manager.register_pass<ov::intel_gpu::ForceFP32Selective>(std::vector<std::string>{},
                                                          std::vector<std::string>{"selected"});
    manager.run_passes(model);
    model->validate_nodes_and_infer_types();
}

size_t count_converts(const std::shared_ptr<ov::Model>& model) {
    size_t count = 0;
    for (const auto& node : model->get_ops()) {
        if (ov::is_type<ov::op::v0::Convert>(node))
            count++;
    }
    return count;
}

void check_filter_selection(const std::string& type_list,
                            const std::string& name_list,
                            unsigned int expected_mask) {
    SCOPED_TRACE(type_list + " / " + name_list);
    const auto types = type_list.empty() ? std::vector<std::string>{}
                                        : ov::Any(type_list).as<std::vector<std::string>>();
    const auto names = name_list.empty() ? std::vector<std::string>{}
                                        : ov::Any(name_list).as<std::vector<std::string>>();
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 4});
    auto alpha = std::make_shared<ov::op::v0::Relu>(input);
    auto beta = std::make_shared<ov::op::v0::Relu>(input);
    auto rms = std::make_shared<ov::op::internal::RMS>(input, 1e-5, ov::element::f16);
    alpha->set_friendly_name("block_alpha");
    beta->set_friendly_name("block_beta");
    rms->set_friendly_name("block_alpha_rms");
    const ov::NodeVector nodes{alpha, beta, rms};
    auto model = std::make_shared<ov::Model>(ov::OutputVector{alpha, beta, rms}, ov::ParameterVector{input});
    const auto original_ops = model->get_ordered_ops();
    ov::intel_gpu::ForceFP32Selective pass(types, names);

    EXPECT_EQ(pass.run_on_model(model), expected_mask != 0u);
    model->validate_nodes_and_infer_types();

    for (size_t port = 0; port < nodes.size(); port++) {
        EXPECT_EQ(nodes[port]->get_output_element_type(0),
                  (expected_mask & (1u << port)) ? ov::element::f32 : ov::element::f16);
        EXPECT_EQ(model->get_output_element_type(port), ov::element::f16);
        if (!(expected_mask & (1u << port))) {
            EXPECT_EQ(nodes[port]->input_value(0), input->output(0));
            EXPECT_EQ(model->get_results()[port]->input_value(0), nodes[port]->output(0));
        }
    }
    EXPECT_EQ(input->get_element_type(), ov::element::f16);
    if (expected_mask == 0u) {
        EXPECT_EQ(count_converts(model), 0u);
        EXPECT_EQ(model->get_ordered_ops(), original_ops);
    }
}

TEST(ForceFP32Selective, FilterListsPreserveOrAndAndNoOpRules) {
    using FilterCase = std::tuple<std::string, std::string, unsigned int>;
    const std::vector<FilterCase> cases{
        {"", "", 0u},
        {"Relu RMS", "", 7u},
        {"", "alpha beta", 7u},
        {"Relu", "alpha", 1u},
        {"Relu", "missing", 0u},
    };
    for (const auto& [type_list, name_list, expected_mask] : cases)
        check_filter_selection(type_list, name_list, expected_mask);
}

TEST(ForceFP32Selective, TypeOnlyFilterMatchesAllNodesOfTheTypeAndUnknownTypeIsNoOp) {
    check_filter_selection("Relu", "", 3u);
    check_filter_selection("AbsentOp", "", 0u);
}

TEST(ForceFP32Selective, NameOnlyFilterMatchesSubstringsAndMissingNameIsNoOp) {
    check_filter_selection("", "alpha", 5u);
    check_filter_selection("", "missing", 0u);
}

TEST(ForceFP32Selective, CombinedListsIntersectAndDisjointMatchesAreNoOp) {
    check_filter_selection("Relu RMS", "beta missing", 2u);
    // Relu matches alpha/beta; the "rms" substring matches only RMS.
    check_filter_selection("Relu", "rms", 0u);
}

TEST(ForceFP32Selective, PrecisionAttributesSurviveAutomaticValidationCloneAndReapplication) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 4});
    auto condition = std::make_shared<ov::op::v0::Parameter>(ov::element::boolean, ov::Shape{1, 4});
    auto weights = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{2, 4}, {1});
    auto bias = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{2}, {0});
    auto rms = std::make_shared<ov::op::internal::RMS>(input, 1e-5, ov::element::f16);
    auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input, weights, bias, ov::element::f16);
    auto select = std::make_shared<ov::op::TypeRelaxed<ov::op::v1::Select>>(
        ov::op::v1::Select(condition, input, input),
        ov::element::TypeVector{ov::element::boolean, ov::element::f16, ov::element::dynamic},
        ov::element::TypeVector{ov::element::f16});
    rms->set_friendly_name("selected_rms");
    fc->set_friendly_name("selected_fc");
    select->set_friendly_name("selected_select");
    auto model = std::make_shared<ov::Model>(ov::OutputVector{rms, fc, select}, ov::ParameterVector{input, condition});
    ov::pass::Manager manager;
    manager.register_pass<ov::intel_gpu::ForceFP32Selective>(std::vector<std::string>{},
                                                          std::vector<std::string>{"selected"});

    manager.run_passes(model);

    for (const auto& output : {rms->output(0), fc->output(0), select->output(0)})
        EXPECT_EQ(output.get_element_type(), ov::element::f32);
    EXPECT_EQ(fc->get_output_type(), ov::element::f32);
    EXPECT_EQ(select->get_origin_input_type(0), ov::element::boolean);
    EXPECT_EQ(select->get_origin_input_type(1), ov::element::f32);
    EXPECT_EQ(select->get_origin_input_type(2), ov::element::dynamic);
    EXPECT_EQ(select->get_overridden_output_type(0), ov::element::f32);
    EXPECT_EQ(rms->clone_with_new_inputs({input})->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(weights->get_element_type(), ov::element::f16);
    EXPECT_EQ(condition->get_element_type(), ov::element::boolean);
    const auto original_ops = model->get_ordered_ops();
    const auto convert_count = count_converts(model);
    ov::intel_gpu::ForceFP32Selective pass({}, {"selected"});
    EXPECT_FALSE(pass.run_on_model(model));
    EXPECT_EQ(model->get_ordered_ops(), original_ops);
    EXPECT_EQ(count_converts(model), convert_count);
}

TEST(ForceFP32Selective, MixedOutputsPromoteOnlyFP16Port) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1});
    auto mixed = std::make_shared<MixedOutputOp>(input);
    auto consumer = std::make_shared<ov::op::v0::Relu>(mixed->output(1));
    mixed->set_friendly_name("selected_mixed");
    consumer->set_friendly_name("selected_consumer");
    input->set_friendly_name("selected_input");
    mixed->output(1).set_names({"mixed_tensor"});
    auto outputs = mixed->outputs();
    outputs.push_back(consumer);
    auto model = std::make_shared<ov::Model>(outputs, ov::ParameterVector{input});

    const auto original_names = model->output(1).get_names();
    model->get_results()[1]->set_friendly_name("selected_result");
    force_selected(model);

    EXPECT_EQ(model->output(1).get_names(), original_names);
    EXPECT_EQ(model->get_results()[1]->get_friendly_name(), "selected_result");
    EXPECT_EQ(input->get_element_type(), ov::element::f16);
    EXPECT_EQ(model->get_output_partial_shape(1), ov::PartialShape{1});
    const ov::element::TypeVector original_types{ov::element::i32,
                                                ov::element::f16,
                                                ov::element::f32,
                                                ov::element::boolean,
                                                ov::element::bf16};
    for (size_t port = 0; port < original_types.size(); port++) {
        EXPECT_EQ(mixed->get_output_element_type(port),
                  port == 1 ? ov::element::f32 : original_types[port]);
        EXPECT_EQ(model->get_output_element_type(port), original_types[port]);
        if (port != 1) {
            EXPECT_EQ(model->get_results()[port]->input_value(0), mixed->output(port));
        }
    }
    EXPECT_EQ(consumer->input_value(0), mixed->output(1));
    EXPECT_EQ(consumer->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(count_converts(model), 3u);
}

TEST(ForceFP32Selective, InternalConnectionDoesNotRoundToFP16) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1});
    auto factor = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1}, {1.0009765625f});
    auto first = std::make_shared<ov::op::v1::Multiply>(input, factor);
    auto down = std::make_shared<ov::op::v0::Convert>(first, ov::element::f16);
    auto up = std::make_shared<ov::op::v0::Convert>(down, ov::element::f32);
    auto rounded = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1}, {1.001953125f});
    auto half = std::make_shared<ov::op::v0::Convert>(up, ov::element::f16);
    auto second = std::make_shared<ov::op::v1::Subtract>(half, rounded);
    first->set_friendly_name("selected_first");
    second->set_friendly_name("selected_second");
    auto model = std::make_shared<ov::Model>(ov::OutputVector{second}, ov::ParameterVector{input});
    ov::Tensor data(ov::element::f16, ov::Shape{1});
    data.data<ov::float16>()[0] = ov::float16(1.0009765625f);
    ov::TensorVector before{ov::Tensor(ov::element::f16, ov::Shape{1})};
    ASSERT_TRUE(model->evaluate(before, {data}));
    EXPECT_EQ(static_cast<float>(before[0].data<ov::float16>()[0]), 0.0f);

    force_selected(model);

    EXPECT_EQ(second->input_value(0).get_node_shared_ptr(), first);
    ov::TensorVector after{ov::Tensor(ov::element::f16, ov::Shape{1})};
    ASSERT_TRUE(model->evaluate(after, {data}));
    EXPECT_EQ(static_cast<float>(after[0].data<ov::float16>()[0]), 0.00000095367431640625f);
}

TEST(ForceFP32Selective, ExternalSharedRoundTripKeepsUnselectedValues) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
    auto down = std::make_shared<ov::op::v0::Convert>(input, ov::element::f16);
    auto offset = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1}, {1});
    auto selected = std::make_shared<ov::op::v1::Subtract>(down, offset);
    auto other = std::make_shared<ov::op::v0::Convert>(down, ov::element::f32);
    selected->set_friendly_name("selected");
    other->set_friendly_name("other");
    auto model = std::make_shared<ov::Model>(ov::OutputVector{selected, other}, ov::ParameterVector{input});
    ov::Tensor data(ov::element::f32, ov::Shape{1});
    data.data<float>()[0] = 1.0001f;
    ov::TensorVector expected{ov::Tensor(ov::element::f16, ov::Shape{1}),
                              ov::Tensor(ov::element::f32, ov::Shape{1})};
    ASSERT_TRUE(model->evaluate(expected, {data}));

    force_selected(model);

    EXPECT_EQ(selected->input_value(0).get_node_shared_ptr(), input);
    EXPECT_EQ(other->input_value(0).get_node_shared_ptr(), down);
    EXPECT_EQ(down->input_value(0).get_node_shared_ptr(), input);
    ov::TensorVector actual{ov::Tensor(ov::element::f16, ov::Shape{1}),
                            ov::Tensor(ov::element::f32, ov::Shape{1})};
    ASSERT_TRUE(model->evaluate(actual, {data}));
    EXPECT_EQ(static_cast<float>(expected[0].data<ov::float16>()[0]), 0.0f);
    EXPECT_EQ(static_cast<float>(actual[0].data<ov::float16>()[0]),
              static_cast<float>(ov::float16(data.data<float>()[0] - 1.0f)));
    EXPECT_GT(static_cast<float>(actual[0].data<ov::float16>()[0]), 0.0f);
    EXPECT_FLOAT_EQ(actual[1].data<float>()[0], expected[1].data<float>()[0]);
    EXPECT_NE(actual[1].data<float>()[0], data.data<float>()[0]);
}

TEST(ForceFP32Selective, PassThroughAliasIsExplicitAndRespectsNameFilter) {
    using AliasCase = std::tuple<std::string, std::string, bool, bool, bool>;
    // Flags: select target arithmetic, middle pass-through ops, other pass-through ops.
    const std::vector<AliasCase> cases{
        {"Multiply Subtract", "", true, false, false},
        {"", "target middle", true, true, false},
        {"Multiply Subtract PASSTHROUGH_OPS", "", true, true, true},
        {"Multiply Subtract PASSTHROUGH_OPS", "target middle", true, true, false},
        {"Multiply Subtract PASSTHROUGH_OPS", "target", true, false, false},
        {"PASSTHROUGH_OPS", "", false, true, true},
    };
    for (const auto& [type_list, name_list, targets_selected, middle_selected, others_selected] : cases) {
        SCOPED_TRACE(type_list + " / " + name_list);
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 4});
        auto factor = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1}, {1.0009765625f});
        auto first = std::make_shared<ov::op::v1::Multiply>(input, factor);
        auto shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {2, 2});
        auto reshape = std::make_shared<ov::op::v1::Reshape>(first, shape, false);
        auto order = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, 0});
        auto transpose = std::make_shared<ov::op::v1::Transpose>(reshape, order);
        auto rounded = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1}, {1.001953125f});
        auto second = std::make_shared<ov::op::v1::Subtract>(transpose, rounded);
        auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
        auto split = std::make_shared<ov::op::v1::Split>(second, axis, 2);
        auto left = std::make_shared<ov::op::v0::Relu>(split->output(0));
        auto right = std::make_shared<ov::op::v0::Relu>(split->output(1));
        auto isolated = std::make_shared<ov::op::v1::Reshape>(input, shape, false);
        auto shape_of = std::make_shared<ov::op::v3::ShapeOf>(second, ov::element::i64);
        first->set_friendly_name("target_first");
        second->set_friendly_name("target_second");
        reshape->set_friendly_name("middle_reshape");
        transpose->set_friendly_name("middle_transpose");
        split->set_friendly_name("other_split");
        isolated->set_friendly_name("other_isolated");
        shape_of->set_friendly_name("middle_shape");
        auto model = std::make_shared<ov::Model>(ov::OutputVector{left, right, isolated, shape_of},
                                               ov::ParameterVector{input});
        const auto types = type_list.empty() ? std::vector<std::string>{}
                                            : ov::Any(type_list).as<std::vector<std::string>>();
        const auto names = name_list.empty() ? std::vector<std::string>{}
                                            : ov::Any(name_list).as<std::vector<std::string>>();
        ov::pass::Manager manager;
        manager.register_pass<ov::intel_gpu::ForceFP32Selective>(types, names);

        manager.run_passes(model);

        for (const auto& node : {first->output(0), second->output(0)})
            EXPECT_EQ(node.get_element_type(), targets_selected ? ov::element::f32 : ov::element::f16);
        EXPECT_EQ(reshape->get_output_element_type(0), middle_selected ? ov::element::f32 : ov::element::f16);
        EXPECT_EQ(transpose->get_output_element_type(0), middle_selected ? ov::element::f32 : ov::element::f16);
        for (const auto& output : split->outputs())
            EXPECT_EQ(output.get_element_type(), others_selected ? ov::element::f32 : ov::element::f16);
        EXPECT_EQ(isolated->get_output_element_type(0), others_selected ? ov::element::f32 : ov::element::f16);
        EXPECT_EQ(shape_of->get_output_element_type(0), ov::element::i64);
        EXPECT_EQ(reshape->input_value(1), shape->output(0));
        EXPECT_EQ(transpose->input_value(1), order->output(0));
        EXPECT_EQ(split->input_value(1), axis->output(0));
        EXPECT_EQ(left->get_output_element_type(0), ov::element::f16);
        EXPECT_EQ(right->get_output_element_type(0), ov::element::f16);
        for (size_t port = 0; port < 3; port++)
            EXPECT_EQ(model->get_output_element_type(port), ov::element::f16);
        EXPECT_EQ(model->get_output_element_type(3), ov::element::i64);
        if (targets_selected && middle_selected) {
            EXPECT_EQ(reshape->input_value(0), first->output(0));
            EXPECT_EQ(transpose->input_value(0), reshape->output(0));
            EXPECT_EQ(second->input_value(0), transpose->output(0));
        } else if (targets_selected) {
            auto down = ov::as_type_ptr<ov::op::v0::Convert>(reshape->input_value(0).get_node_shared_ptr());
            auto up = ov::as_type_ptr<ov::op::v0::Convert>(second->input_value(0).get_node_shared_ptr());
            ASSERT_NE(down, nullptr);
            ASSERT_NE(up, nullptr);
            EXPECT_EQ(down->get_destination_type(), ov::element::f16);
            EXPECT_EQ(down->input_value(0), first->output(0));
            EXPECT_EQ(up->get_destination_type(), ov::element::f32);
            EXPECT_EQ(up->input_value(0), transpose->output(0));
        }

        ov::Tensor data(ov::element::f16, ov::Shape{1, 4});
        for (size_t index = 0; index < data.get_size(); index++)
            data.data<ov::float16>()[index] = ov::float16(1.0009765625f);
        ov::TensorVector outputs{ov::Tensor(ov::element::f16, ov::Shape{2, 1}),
                                 ov::Tensor(ov::element::f16, ov::Shape{2, 1}),
                                 ov::Tensor(ov::element::f16, ov::Shape{2, 2}),
                                 ov::Tensor(ov::element::i64, ov::Shape{2})};
        ASSERT_TRUE(model->evaluate(outputs, {data}));
        const float expected = targets_selected && middle_selected ? 0.00000095367431640625f : 0.0f;
        for (size_t port = 0; port < 2; port++) {
            for (size_t index = 0; index < outputs[port].get_size(); index++)
                EXPECT_EQ(static_cast<float>(outputs[port].data<ov::float16>()[index]), expected);
        }
    }
}

TEST(ForceFP32Selective, RenamedPlaceholderRemainsAnAbsentOptionalInput) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 4});
    auto weights = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{2, 4}, {1});
    auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
    auto producer = std::make_shared<ov::op::v0::Relu>(input);
    producer->set_friendly_name("input_Placeholder");
    auto selected = std::make_shared<ov::intel_gpu::op::FullyConnected>(producer,
                                                                      weights,
                                                                      no_bias,
                                                                      ov::element::f16);
    no_bias->set_friendly_name("selected_optional_input");
    selected->set_friendly_name("selected_Placeholder_fc");
    auto model = std::make_shared<ov::Model>(ov::OutputVector{selected}, ov::ParameterVector{input});

    force_selected(model);

    EXPECT_EQ(selected->input_value(2).get_node_shared_ptr(), no_bias);
    EXPECT_EQ(no_bias->get_output_element_type(0), ov::element::dynamic);
    EXPECT_EQ(no_bias->get_friendly_name(), "selected_optional_input");
    EXPECT_EQ(selected->get_input_element_type(2), ov::element::dynamic);
    EXPECT_EQ(selected->get_output_element_type(0), ov::element::f32);
    auto promote = ov::as_type_ptr<ov::op::v0::Convert>(selected->input_value(0).get_node_shared_ptr());
    ASSERT_NE(promote, nullptr);
    EXPECT_EQ(promote->input_value(0), producer->output(0));
    EXPECT_EQ(producer->get_output_element_type(0), ov::element::f16);
    EXPECT_EQ(count_converts(model), 3u);
}

}  // namespace

#endif  // GPU_DEBUG_CONFIG
