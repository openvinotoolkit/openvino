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
#include "intel_gpu/op/placeholder.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/sink.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/op/select.hpp"
#include "openvino/pass/manager.hpp"
#include "ov_ops/rms.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "plugin/transformations/force_fp32_selective.hpp"

namespace force_fp32_selective_test {

// Mixed outputs exercise an FP16 port beyond output 0 and unrelated output types.
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

class OutputlessSink : public ov::op::Sink {
public:
    OPENVINO_OP("OutputlessSink", "test", ov::op::Sink);

    explicit OutputlessSink(const ov::Output<ov::Node>& input) {
        set_arguments({input});
    }

    void validate_and_infer_types() override {}

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return std::make_shared<OutputlessSink>(inputs.at(0));
    }
};

}  // namespace force_fp32_selective_test

namespace {

using force_fp32_selective_test::MixedOutputOp;
using force_fp32_selective_test::OutputlessSink;

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

TEST(ForceFP32Selective, OrdinaryNodeWithPlaceholderInNameIsPromoted) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 4});
    auto selected = std::make_shared<ov::op::v0::Relu>(input);
    selected->set_friendly_name("selected_Placeholder_layer");
    auto model = std::make_shared<ov::Model>(ov::OutputVector{selected}, ov::ParameterVector{input});

    force_selected(model);

    EXPECT_EQ(selected->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(selected->get_input_element_type(0), ov::element::f32);
    EXPECT_EQ(selected->get_friendly_name(), "selected_Placeholder_layer");
    EXPECT_EQ(model->get_output_element_type(0), ov::element::f16);
    EXPECT_EQ(count_converts(model), 2u);
}

TEST(ForceFP32Selective, OrdinaryInputWithPlaceholderInNameIsConverted) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 4});
    auto producer = std::make_shared<ov::op::v0::Relu>(input);
    auto selected = std::make_shared<ov::op::v0::Relu>(producer);
    producer->set_friendly_name("Placeholder_producer");
    selected->set_friendly_name("selected");
    auto model = std::make_shared<ov::Model>(ov::OutputVector{selected}, ov::ParameterVector{input});

    force_selected(model);

    auto promote = ov::as_type_ptr<ov::op::v0::Convert>(selected->input_value(0).get_node_shared_ptr());
    ASSERT_NE(promote, nullptr);
    EXPECT_EQ(promote->input_value(0).get_node_shared_ptr(), producer);
    EXPECT_EQ(promote->get_destination_type(), ov::element::f32);
    EXPECT_EQ(producer->get_output_element_type(0), ov::element::f16);
    EXPECT_EQ(selected->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(count_converts(model), 2u);
}

TEST(ForceFP32Selective, RenamedPlaceholderRemainsAnAbsentOptionalInput) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 4});
    auto weights = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{2, 4}, {1});
    auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
    auto selected = std::make_shared<ov::intel_gpu::op::FullyConnected>(input,
                                                                      weights,
                                                                      no_bias,
                                                                      ov::element::f16);
    no_bias->set_friendly_name("selected_optional_input");
    selected->set_friendly_name("selected_fc");
    auto model = std::make_shared<ov::Model>(ov::OutputVector{selected}, ov::ParameterVector{input});

    force_selected(model);

    EXPECT_EQ(selected->input_value(2).get_node_shared_ptr(), no_bias);
    EXPECT_EQ(no_bias->get_output_element_type(0), ov::element::dynamic);
    EXPECT_EQ(no_bias->get_friendly_name(), "selected_optional_input");
    EXPECT_EQ(selected->get_input_element_type(2), ov::element::dynamic);
    EXPECT_EQ(selected->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(count_converts(model), 3u);
}

TEST(ForceFP32Selective, SplitPromotesAndRestoresEveryFP16Output) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 4});
    auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
    auto split = std::make_shared<ov::op::v1::Split>(input, axis, 2);
    split->set_friendly_name("selected_split");
    split->output(0).set_names({"left"});
    split->output(1).set_names({"right"});
    auto model = std::make_shared<ov::Model>(ov::OutputVector{split->output(0), split->output(1)},
                                           ov::ParameterVector{input});
    const auto left_names = model->output(0).get_names();
    const auto right_names = model->output(1).get_names();

    force_selected(model);

    for (size_t port = 0; port < split->get_output_size(); port++) {
        EXPECT_EQ(split->get_output_element_type(port), ov::element::f32);
        EXPECT_EQ(model->get_output_element_type(port), ov::element::f16);
        EXPECT_EQ(model->get_output_partial_shape(port), (ov::PartialShape{1, 2}));
        auto restore = ov::as_type_ptr<ov::op::v0::Convert>(model->get_results()[port]->input_value(0)
                                                            .get_node_shared_ptr());
        ASSERT_NE(restore, nullptr);
        EXPECT_EQ(restore->input_value(0), split->output(port));
    }
    EXPECT_EQ(model->output(0).get_names(), left_names);
    EXPECT_EQ(model->output(1).get_names(), right_names);
    EXPECT_EQ(split->input_value(1).get_node_shared_ptr(), axis);
    EXPECT_EQ(axis->get_element_type(), ov::element::i64);
    EXPECT_EQ(count_converts(model), 3u);
}

TEST(ForceFP32Selective, SplitSecondOutputConnectsDirectlyWithinRegion) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 4});
    auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
    auto split = std::make_shared<ov::op::v1::Split>(input, axis, 2);
    auto down = std::make_shared<ov::op::v0::Convert>(split->output(1), ov::element::f16);
    auto up = std::make_shared<ov::op::v0::Convert>(down, ov::element::f32);
    auto consumer = std::make_shared<ov::op::internal::RMS>(up, 1e-5, ov::element::f16);
    split->set_friendly_name("selected_split");
    consumer->set_friendly_name("selected_consumer");
    auto model = std::make_shared<ov::Model>(ov::OutputVector{split->output(0), consumer},
                                           ov::ParameterVector{input});

    force_selected(model);

    EXPECT_EQ(consumer->input_value(0), split->output(1));
    EXPECT_EQ(split->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(split->get_output_element_type(1), ov::element::f32);
    EXPECT_EQ(consumer->get_input_element_type(0), ov::element::f32);
    EXPECT_EQ(count_converts(model), 3u);
}

class ForceFP32SelectiveTopK : public testing::TestWithParam<ov::element::Type> {};

TEST_P(ForceFP32SelectiveTopK, KeepsIntegerIndicesAndRestoresOnlyValues) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 4});
    auto k = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {2});
    auto topk = std::make_shared<ov::op::v3::TopK>(input, k, 1, "max", "value", GetParam());
    auto indices = std::make_shared<ov::op::v0::Relu>(topk->output(1));
    topk->set_friendly_name("selected_topk");
    indices->set_friendly_name("selected_indices");
    topk->output(1).set_names({"indices"});
    auto model = std::make_shared<ov::Model>(ov::OutputVector{topk->output(0), topk->output(1), indices},
                                           ov::ParameterVector{input});
    const auto index_names = model->output(1).get_names();

    force_selected(model);

    EXPECT_EQ(topk->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(topk->get_output_element_type(1), GetParam());
    EXPECT_EQ(indices->get_output_element_type(0), GetParam());
    EXPECT_EQ(indices->input_value(0), topk->output(1));
    EXPECT_EQ(model->get_output_element_type(0), ov::element::f16);
    EXPECT_EQ(model->get_output_element_type(1), GetParam());
    EXPECT_EQ(model->get_results()[1]->input_value(0), topk->output(1));
    EXPECT_EQ(model->output(1).get_names(), index_names);
    EXPECT_EQ(topk->input_value(1).get_node_shared_ptr(), k);
    EXPECT_EQ(count_converts(model), 2u);
}

INSTANTIATE_TEST_SUITE_P(IndexTypes,
                         ForceFP32SelectiveTopK,
                         testing::Values(ov::element::i32, ov::element::i64));

TEST(ForceFP32Selective, MixedOutputsPromoteOnlyFP16Port) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1});
    auto mixed = std::make_shared<MixedOutputOp>(input);
    auto consumer = std::make_shared<ov::op::v0::Relu>(mixed->output(1));
    mixed->set_friendly_name("selected_mixed");
    consumer->set_friendly_name("selected_consumer");
    auto outputs = mixed->outputs();
    outputs.push_back(consumer);
    auto model = std::make_shared<ov::Model>(outputs, ov::ParameterVector{input});

    force_selected(model);

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

TEST(ForceFP32Selective, SelectedBoundaryNamesKeepModelInterface) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 4});
    auto relu = std::make_shared<ov::op::v0::Relu>(input);
    auto result = std::make_shared<ov::op::v0::Result>(relu);
    input->set_friendly_name("selected_input");
    relu->set_friendly_name("selected_relu");
    result->set_friendly_name("selected_result");
    input->output(0).set_names({"input_tensor"});
    result->output(0).set_names({"output_tensor"});
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input});
    const auto input_names = model->input().get_names();
    const auto output_names = model->output().get_names();

    force_selected(model);

    EXPECT_EQ(model->get_parameters()[0], input);
    EXPECT_EQ(model->get_results()[0], result);
    EXPECT_EQ(model->input().get_element_type(), ov::element::f16);
    EXPECT_EQ(model->output().get_element_type(), ov::element::f16);
    EXPECT_EQ(model->input().get_partial_shape(), (ov::PartialShape{1, 4}));
    EXPECT_EQ(model->output().get_partial_shape(), (ov::PartialShape{1, 4}));
    EXPECT_EQ(model->input().get_names(), input_names);
    EXPECT_EQ(model->output().get_names(), output_names);
    EXPECT_EQ(input->get_friendly_name(), "selected_input");
    EXPECT_EQ(result->get_friendly_name(), "selected_result");
    EXPECT_EQ(relu->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(count_converts(model), 2u);
}

TEST(ForceFP32Selective, OutputlessSinkAndBoundaryOnlySelectionAreNoOps) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1});
    auto result = std::make_shared<ov::op::v0::Result>(input);
    auto sink = std::make_shared<OutputlessSink>(input);
    input->set_friendly_name("selected_input");
    result->set_friendly_name("selected_result");
    sink->set_friendly_name("selected_sink");
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result},
                                           ov::SinkVector{sink},
                                           ov::ParameterVector{input});
    ov::intel_gpu::ForceFP32Selective pass({}, {"selected"});

    EXPECT_FALSE(pass.run_on_model(model));
    model->validate_nodes_and_infer_types();

    EXPECT_EQ(sink->get_output_size(), 0u);
    EXPECT_EQ(sink->input_value(0).get_node_shared_ptr(), input);
    EXPECT_EQ(result->input_value(0).get_node_shared_ptr(), input);
    EXPECT_EQ(input->get_element_type(), ov::element::f16);
    EXPECT_EQ(count_converts(model), 0u);
}

TEST(ForceFP32Selective, NonFP16NodesRemainUnchanged) {
    const ov::element::TypeVector types{ov::element::i32, ov::element::boolean,
                                       ov::element::bf16, ov::element::f32};
    ov::ParameterVector inputs;
    ov::OutputVector outputs;
    for (const auto& type : types) {
        auto input = std::make_shared<ov::op::v0::Parameter>(type, ov::Shape{1});
        auto convert = std::make_shared<ov::op::v0::Convert>(input, type);
        convert->set_friendly_name("selected_" + type.to_string());
        inputs.push_back(input);
        outputs.push_back(convert);
    }
    auto model = std::make_shared<ov::Model>(outputs, inputs);
    ov::intel_gpu::ForceFP32Selective pass({}, {"selected"});

    EXPECT_FALSE(pass.run_on_model(model));
    model->validate_nodes_and_infer_types();

    for (size_t port = 0; port < types.size(); port++) {
        EXPECT_EQ(outputs[port].get_element_type(), types[port]);
        EXPECT_EQ(model->get_results()[port]->input_value(0), outputs[port]);
    }
    EXPECT_EQ(count_converts(model), types.size());
}

TEST(ForceFP32Selective, AdjacentSelectedNodesHaveOnlyBoundaryConverts) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 4});
    auto first = std::make_shared<ov::op::v0::Relu>(input);
    auto second = std::make_shared<ov::op::v0::Relu>(first);
    auto third = std::make_shared<ov::op::v0::Relu>(second);
    first->set_friendly_name("selected_first");
    second->set_friendly_name("selected_second");
    third->set_friendly_name("selected_third");
    auto model = std::make_shared<ov::Model>(ov::OutputVector{third}, ov::ParameterVector{input});

    force_selected(model);

    EXPECT_EQ(second->input_value(0).get_node_shared_ptr(), first);
    EXPECT_EQ(third->input_value(0).get_node_shared_ptr(), second);
    EXPECT_EQ(first->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(second->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(third->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(input->get_element_type(), ov::element::f16);
    EXPECT_EQ(model->get_output_element_type(0), ov::element::f16);
    EXPECT_EQ(count_converts(model), 2u);
}

TEST(ForceFP32Selective, PassThroughChainStaysFP32WithoutConvertingShapeInputs) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 4});
    auto first = std::make_shared<ov::op::v0::Relu>(input);
    auto shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {2, 2});
    auto reshape = std::make_shared<ov::op::v1::Reshape>(first, shape, false);
    auto order = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, 0});
    auto transpose = std::make_shared<ov::op::v1::Transpose>(reshape, order);
    auto second = std::make_shared<ov::op::v0::Relu>(transpose);
    first->set_friendly_name("selected_first");
    second->set_friendly_name("selected_second");
    auto model = std::make_shared<ov::Model>(ov::OutputVector{second}, ov::ParameterVector{input});

    force_selected(model);

    EXPECT_EQ(reshape->input_value(0).get_node_shared_ptr(), first);
    EXPECT_EQ(transpose->input_value(0).get_node_shared_ptr(), reshape);
    EXPECT_EQ(second->input_value(0).get_node_shared_ptr(), transpose);
    EXPECT_EQ(reshape->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(transpose->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(reshape->input_value(1).get_node_shared_ptr(), shape);
    EXPECT_EQ(transpose->input_value(1).get_node_shared_ptr(), order);
    EXPECT_EQ(shape->get_element_type(), ov::element::i64);
    EXPECT_EQ(count_converts(model), 2u);
}

TEST(ForceFP32Selective, SelectedConvertRemainsInRegionWithItsConsumer) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4});
    auto convert = std::make_shared<ov::op::v0::Convert>(input, ov::element::f16);
    auto consumer = std::make_shared<ov::op::v0::Relu>(convert);
    convert->set_friendly_name("selected_convert");
    consumer->set_friendly_name("selected_consumer");
    auto model = std::make_shared<ov::Model>(ov::OutputVector{consumer}, ov::ParameterVector{input});

    force_selected(model);

    EXPECT_EQ(consumer->input_value(0).get_node_shared_ptr(), convert);
    EXPECT_EQ(convert->get_friendly_name(), "selected_convert");
    EXPECT_EQ(convert->get_destination_type(), ov::element::f32);
    EXPECT_EQ(consumer->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(model->get_output_element_type(0), ov::element::f16);
    EXPECT_EQ(count_converts(model), 2u);
}

TEST(ForceFP32Selective, InternalConvertChainIsBypassedOnlyForSelectedConsumer) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 4});
    auto first = std::make_shared<ov::op::v0::Relu>(input);
    auto down = std::make_shared<ov::op::v0::Convert>(first, ov::element::f16);
    auto up = std::make_shared<ov::op::v0::Convert>(down, ov::element::f32);
    auto down_again = std::make_shared<ov::op::v0::Convert>(up, ov::element::f16);
    auto second = std::make_shared<ov::op::v0::Relu>(down_again);
    auto other = std::make_shared<ov::op::v0::Relu>(down_again);
    first->set_friendly_name("selected_first");
    second->set_friendly_name("selected_second");
    other->set_friendly_name("other");
    auto model = std::make_shared<ov::Model>(ov::OutputVector{second, other}, ov::ParameterVector{input});

    force_selected(model);

    EXPECT_EQ(second->input_value(0).get_node_shared_ptr(), first);
    EXPECT_EQ(second->get_input_element_type(0), ov::element::f32);
    EXPECT_EQ(down->input_value(0).get_node_shared_ptr(), first);
    EXPECT_EQ(other->input_value(0).get_node_shared_ptr(), down_again);
    EXPECT_EQ(down_again->input_value(0).get_node_shared_ptr(), up);
    EXPECT_EQ(up->input_value(0).get_node_shared_ptr(), down);
    EXPECT_EQ(down->get_destination_type(), ov::element::f16);
    EXPECT_EQ(up->get_destination_type(), ov::element::f32);
    EXPECT_EQ(other->get_output_element_type(0), ov::element::f16);
    EXPECT_EQ(count_converts(model), 5u);
}

TEST(ForceFP32Selective, InternalRoundTripEndingInFP32IsBypassed) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 4});
    auto first = std::make_shared<ov::op::v0::Relu>(input);
    auto down = std::make_shared<ov::op::v0::Convert>(first, ov::element::f16);
    auto up = std::make_shared<ov::op::v0::Convert>(down, ov::element::f32);
    auto second = std::make_shared<ov::op::internal::RMS>(up, 1e-5, ov::element::f16);
    first->set_friendly_name("selected_first");
    second->set_friendly_name("selected_second");
    auto model = std::make_shared<ov::Model>(ov::OutputVector{second}, ov::ParameterVector{input});

    force_selected(model);

    EXPECT_EQ(second->input_value(0).get_node_shared_ptr(), first);
    EXPECT_EQ(second->get_input_element_type(0), ov::element::f32);
    EXPECT_EQ(second->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(count_converts(model), 2u);
}

TEST(ForceFP32Selective, DirectSharedOutputRestoresOnlyUnselectedBranch) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 4});
    auto first = std::make_shared<ov::op::v0::Relu>(input);
    auto second = std::make_shared<ov::op::v0::Relu>(first);
    auto other = std::make_shared<ov::op::v0::Relu>(first);
    first->set_friendly_name("selected_first");
    second->set_friendly_name("selected_second");
    other->set_friendly_name("other");
    auto model = std::make_shared<ov::Model>(ov::OutputVector{second, other}, ov::ParameterVector{input});

    force_selected(model);

    EXPECT_EQ(second->input_value(0).get_node_shared_ptr(), first);
    auto restore = ov::as_type_ptr<ov::op::v0::Convert>(other->input_value(0).get_node_shared_ptr());
    ASSERT_NE(restore, nullptr);
    EXPECT_EQ(restore->input_value(0).get_node_shared_ptr(), first);
    EXPECT_EQ(restore->get_destination_type(), ov::element::f16);
    EXPECT_EQ(other->get_output_element_type(0), ov::element::f16);
    EXPECT_EQ(count_converts(model), 3u);
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

TEST(ForceFP32Selective, ExternalSingleUseConvertIsBypassed) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
    auto down = std::make_shared<ov::op::v0::Convert>(input, ov::element::f16);
    auto selected = std::make_shared<ov::op::v0::Relu>(down);
    down->set_friendly_name("original_downcast");
    selected->set_friendly_name("selected");
    auto model = std::make_shared<ov::Model>(ov::OutputVector{selected}, ov::ParameterVector{input});

    force_selected(model);

    EXPECT_EQ(selected->input_value(0).get_node_shared_ptr(), input);
    EXPECT_EQ(selected->get_input_element_type(0), ov::element::f32);
    EXPECT_EQ(count_converts(model), 1u);
    EXPECT_EQ(down->input_value(0).get_node_shared_ptr(), input);
    EXPECT_EQ(down->get_destination_type(), ov::element::f16);
    EXPECT_EQ(down->get_friendly_name(), "original_downcast");
}

TEST(ForceFP32Selective, ExternalFP32ConvertChainKeepsSharedConsumers) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4});
    auto down = std::make_shared<ov::op::v0::Convert>(input, ov::element::f16);
    auto up = std::make_shared<ov::op::v0::Convert>(down, ov::element::f32);
    auto half = std::make_shared<ov::op::v0::Convert>(up, ov::element::f16);
    auto selected = std::make_shared<ov::op::v0::Relu>(half);
    auto other = std::make_shared<ov::op::v0::Relu>(half);
    selected->set_friendly_name("selected");
    other->set_friendly_name("other");
    auto model = std::make_shared<ov::Model>(ov::OutputVector{selected, other}, ov::ParameterVector{input});

    force_selected(model);

    EXPECT_EQ(selected->input_value(0).get_node_shared_ptr(), input);
    EXPECT_EQ(other->input_value(0).get_node_shared_ptr(), half);
    EXPECT_EQ(half->input_value(0).get_node_shared_ptr(), up);
    EXPECT_EQ(up->input_value(0).get_node_shared_ptr(), down);
    EXPECT_EQ(down->input_value(0).get_node_shared_ptr(), input);
    EXPECT_EQ(other->get_output_element_type(0), ov::element::f16);
    EXPECT_EQ(count_converts(model), 4u);
}

TEST(ForceFP32Selective, ExternalRoundTripEndingInFP32IsBypassed) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4});
    auto down = std::make_shared<ov::op::v0::Convert>(input, ov::element::f16);
    auto up = std::make_shared<ov::op::v0::Convert>(down, ov::element::f32);
    auto selected = std::make_shared<ov::op::internal::RMS>(up, 1e-5, ov::element::f16);
    auto other = std::make_shared<ov::op::v0::Relu>(up);
    selected->set_friendly_name("selected");
    other->set_friendly_name("other");
    auto model = std::make_shared<ov::Model>(ov::OutputVector{selected, other}, ov::ParameterVector{input});

    force_selected(model);

    EXPECT_EQ(selected->input_value(0).get_node_shared_ptr(), input);
    EXPECT_EQ(other->input_value(0).get_node_shared_ptr(), up);
    EXPECT_EQ(up->input_value(0).get_node_shared_ptr(), down);
    EXPECT_EQ(down->get_destination_type(), ov::element::f16);
    EXPECT_EQ(other->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(count_converts(model), 3u);
}

TEST(ForceFP32Selective, ExternalFP16ProducerKeepsItsConvertChain) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1});
    auto up = std::make_shared<ov::op::v0::Convert>(input, ov::element::f32);
    auto down = std::make_shared<ov::op::v0::Convert>(up, ov::element::f16);
    auto selected = std::make_shared<ov::op::v0::Relu>(down);
    selected->set_friendly_name("selected");
    auto model = std::make_shared<ov::Model>(ov::OutputVector{selected}, ov::ParameterVector{input});

    force_selected(model);

    auto promote = ov::as_type_ptr<ov::op::v0::Convert>(selected->input_value(0).get_node_shared_ptr());
    ASSERT_NE(promote, nullptr);
    EXPECT_EQ(promote->input_value(0).get_node_shared_ptr(), down);
    EXPECT_EQ(down->input_value(0).get_node_shared_ptr(), up);
    EXPECT_EQ(up->input_value(0).get_node_shared_ptr(), input);
    EXPECT_EQ(input->get_element_type(), ov::element::f16);
    EXPECT_EQ(count_converts(model), 4u);
}

class ForceFP32SelectiveExternalConversions : public testing::TestWithParam<ov::element::Type> {};

TEST_P(ForceFP32SelectiveExternalConversions, KeepsConversionsThroughOtherTypes) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
    auto narrow = std::make_shared<ov::op::v0::Convert>(input, GetParam());
    auto half = std::make_shared<ov::op::v0::Convert>(narrow, ov::element::f16);
    auto selected = std::make_shared<ov::op::v0::Relu>(half);
    selected->set_friendly_name("selected");
    auto model = std::make_shared<ov::Model>(ov::OutputVector{selected}, ov::ParameterVector{input});

    force_selected(model);

    auto promote = ov::as_type_ptr<ov::op::v0::Convert>(selected->input_value(0).get_node_shared_ptr());
    ASSERT_NE(promote, nullptr);
    EXPECT_EQ(promote->input_value(0).get_node_shared_ptr(), half);
    EXPECT_EQ(half->input_value(0).get_node_shared_ptr(), narrow);
    EXPECT_EQ(narrow->input_value(0).get_node_shared_ptr(), input);
    EXPECT_EQ(narrow->get_destination_type(), GetParam());
    EXPECT_EQ(count_converts(model), 4u);
}

INSTANTIATE_TEST_SUITE_P(SourceTypes,
                         ForceFP32SelectiveExternalConversions,
                         testing::Values(ov::element::i32, ov::element::boolean, ov::element::bf16));

TEST(ForceFP32Selective, IntegerConvertChainIsNotBypassed) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1});
    auto first = std::make_shared<ov::op::v0::Relu>(input);
    auto to_integer = std::make_shared<ov::op::v0::Convert>(first, ov::element::i32);
    auto to_half = std::make_shared<ov::op::v0::Convert>(to_integer, ov::element::f16);
    auto second = std::make_shared<ov::op::v0::Relu>(to_half);
    first->set_friendly_name("selected_first");
    second->set_friendly_name("selected_second");
    auto model = std::make_shared<ov::Model>(ov::OutputVector{second}, ov::ParameterVector{input});

    force_selected(model);

    auto promote = ov::as_type_ptr<ov::op::v0::Convert>(second->input_value(0).get_node_shared_ptr());
    ASSERT_NE(promote, nullptr);
    EXPECT_EQ(promote->input_value(0).get_node_shared_ptr(), to_half);
    EXPECT_EQ(to_half->input_value(0).get_node_shared_ptr(), to_integer);
    EXPECT_EQ(to_integer->get_destination_type(), ov::element::i32);
    EXPECT_EQ(to_integer->get_input_element_type(0), ov::element::f16);
    EXPECT_EQ(second->get_output_element_type(0), ov::element::f32);
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
