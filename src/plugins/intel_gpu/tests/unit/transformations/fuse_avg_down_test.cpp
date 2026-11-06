// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/transformations/fuse_avg_down.hpp"

#include <gtest/gtest.h>

#include "intel_gpu/op/grouped_space_to_depth.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/floor_mod.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/manager.hpp"

namespace {

std::shared_ptr<ov::op::v0::Constant> shape_part(const std::vector<int64_t>& values) {
    return ov::op::v0::Constant::create(ov::element::i64, ov::Shape{values.size()}, values);
}

std::shared_ptr<ov::op::v0::Constant> scalar(int64_t value) {
    return ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {value});
}

enum class DynamicPaddingVariant { Canonical, MisplacedTemporal, NonzeroSpatial };

std::shared_ptr<ov::Model> make_avg_down_model(int64_t input_channels,
                                               int64_t output_channels,
                                               int64_t factor_t,
                                               int64_t factor_s,
                                               bool add_pad,
                                               bool valid_transpose = true,
                                               bool add_consumer = false,
                                               bool dynamic_temporal_pad = false,
                                               DynamicPaddingVariant padding_variant = DynamicPaddingVariant::Canonical) {
    const int64_t input_time = 5;
    const int64_t input_height = 8;
    const int64_t input_width = 10;
    const int64_t pad_begin_t = (factor_t - input_time % factor_t) % factor_t;
    const int64_t output_time = (input_time + pad_begin_t) / factor_t;
    const int64_t output_height = input_height / factor_s;
    const int64_t output_width = input_width / factor_s;
    const int64_t factor_volume = factor_t * factor_s * factor_s;
    const int64_t group_size = input_channels * factor_volume / output_channels;

    auto parameter = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                             ov::PartialShape{dynamic_temporal_pad ? ov::Dimension::dynamic() : ov::Dimension(1),
                                                                              input_channels,
                                                                              dynamic_temporal_pad ? ov::Dimension::dynamic() : ov::Dimension(input_time),
                                                                              dynamic_temporal_pad ? ov::Dimension::dynamic() : ov::Dimension(input_height),
                                                                              dynamic_temporal_pad ? ov::Dimension::dynamic() : ov::Dimension(input_width)});
    ov::Output<ov::Node> data = parameter;
    if (add_pad) {
        ov::Output<ov::Node> pads_begin = shape_part({0, 0, pad_begin_t, 0, 0});
        if (dynamic_temporal_pad) {
            auto shape_of = std::make_shared<ov::op::v3::ShapeOf>(data, ov::element::i64);
            auto time_length = std::make_shared<ov::op::v8::Gather>(shape_of, shape_part({2}), scalar(0));
            auto inner_mod = std::make_shared<ov::op::v1::FloorMod>(time_length, shape_part({factor_t}));
            auto negated_mod = std::make_shared<ov::op::v1::Multiply>(inner_mod, shape_part({-1}));
            auto subtract = std::make_shared<ov::op::v1::Add>(shape_part({factor_t}), negated_mod);
            auto outer_mod = std::make_shared<ov::op::v1::FloorMod>(subtract, shape_part({factor_t}));
            ov::OutputVector padding_parts =
                {shape_part({0}), shape_part({0}), shape_part({0}), shape_part({0}), outer_mod, shape_part({0})};
            if (padding_variant == DynamicPaddingVariant::MisplacedTemporal) {
                std::swap(padding_parts[0], padding_parts[4]);
            } else if (padding_variant == DynamicPaddingVariant::NonzeroSpatial) {
                padding_parts[0] = shape_part({1});
            }
            auto padding_values = std::make_shared<ov::op::v0::Concat>(padding_parts, 0);
            auto paired_padding = std::make_shared<ov::op::v1::Reshape>(padding_values, shape_part({-1, 2}), false);
            auto split_padding = std::make_shared<ov::op::v1::Split>(paired_padding, scalar(1), 2);
            auto first_column = std::make_shared<ov::op::v0::Squeeze>(split_padding->output(0), shape_part({1}));
            auto reversed_padding = std::make_shared<ov::op::v8::Gather>(first_column, shape_part({2, 1, 0}), scalar(0));
            pads_begin = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{shape_part({0, 0}), reversed_padding}, 0);
        }
        data = std::make_shared<ov::op::v12::Pad>(data,
                                                  pads_begin,
                                                  shape_part({0, 0, 0, 0, 0}),
                                                  ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {0}),
                                                  ov::op::PadMode::CONSTANT);
    }

    auto batch_channels = shape_part({1, input_channels});
    auto batch = shape_part({1});
    auto time = shape_part({output_time});
    auto height = shape_part({output_height});
    auto width = shape_part({output_width});
    auto factor_shape = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector{batch_channels, time, shape_part({factor_t}), height, shape_part({factor_s}), width, shape_part({factor_s})},
        0);
    auto factor_reshape = std::make_shared<ov::op::v1::Reshape>(data, factor_shape, false);
    auto order = valid_transpose ? std::vector<int64_t>{0, 1, 3, 5, 7, 2, 4, 6} : std::vector<int64_t>{0, 1, 3, 5, 6, 2, 4, 7};
    auto transpose = std::make_shared<ov::op::v1::Transpose>(factor_reshape, shape_part(order));

    ov::OutputVector flatten_parts;
    if (factor_volume == 1) {
        flatten_parts = {batch_channels, time, height, width};
    } else {
        flatten_parts = {batch, shape_part({input_channels * factor_volume}), time, height, width};
    }
    auto flatten_shape = std::make_shared<ov::op::v0::Concat>(flatten_parts, 0);
    auto flatten_reshape = std::make_shared<ov::op::v1::Reshape>(transpose, flatten_shape, false);
    auto group_shape =
        std::make_shared<ov::op::v0::Concat>(ov::OutputVector{batch, shape_part({output_channels}), shape_part({group_size}), time, height, width}, 0);
    auto group_reshape = std::make_shared<ov::op::v1::Reshape>(flatten_reshape, group_shape, false);
    auto reduce = std::make_shared<ov::op::v1::ReduceMean>(group_reshape, shape_part({2}), false);
    ov::Output<ov::Node> output = reduce;
    if (add_consumer) {
        output = std::make_shared<ov::op::v0::Relu>(output);
    }
    return std::make_shared<ov::Model>(ov::OutputVector{output}, ov::ParameterVector{parameter});
}

void run_pass(const std::shared_ptr<ov::Model>& model) {
    ov::pass::Manager manager;
    manager.register_pass<ov::intel_gpu::FuseAvgDown>();
    manager.run_passes(model);
}

std::shared_ptr<ov::op::v12::Pad> find_pad(const std::shared_ptr<ov::Model>& model) {
    for (const auto& node : model->get_ops()) {
        if (const auto pad = ov::as_type_ptr<ov::op::v12::Pad>(node)) {
            return pad;
        }
    }
    return {};
}

TEST(FuseAvgDownTest, RewritesSpatialAverageToAvgPool3D) {
    auto model = make_avg_down_model(96, 96, 1, 2, false);
    run_pass(model);

    EXPECT_EQ(model->get_results().front()->get_input_node_ptr(0)->get_type_name(), std::string("AvgPool"));
}

TEST(FuseAvgDownTest, RemovesUnitFactorChainAndPadding) {
    auto model = make_avg_down_model(768, 768, 1, 1, true, true, true);
    run_pass(model);

    const auto consumer = model->get_results().front()->get_input_node_shared_ptr(0);
    ASSERT_EQ(consumer->get_type_name(), std::string("Relu"));
    EXPECT_EQ(consumer->get_input_node_ptr(0)->get_type_name(), std::string("Parameter"));
}

TEST(FuseAvgDownTest, RewritesAllTemporalGroupedBlocksToGroupedSpaceToDepth) {
    const std::vector<std::pair<int64_t, int64_t>> channel_transitions = {
        {96, 192},
        {192, 384},
        {384, 768},
    };
    for (const auto& [input_channels, output_channels] : channel_transitions) {
        SCOPED_TRACE(input_channels);
        auto model = make_avg_down_model(input_channels, output_channels, 2, 2, true);
        run_pass(model);

        const auto grouped_space_to_depth = ov::as_type_ptr<ov::intel_gpu::op::GroupedSpaceToDepth>(model->get_results().front()->get_input_node_shared_ptr(0));
        ASSERT_TRUE(grouped_space_to_depth);
        EXPECT_EQ(grouped_space_to_depth->get_factor_t(), 2);
        EXPECT_EQ(grouped_space_to_depth->get_factor_s(), 2);
        EXPECT_EQ(grouped_space_to_depth->get_output_channels(), static_cast<size_t>(output_channels));
        EXPECT_EQ(grouped_space_to_depth->get_output_shape(0), ov::Shape({1, static_cast<size_t>(output_channels), 3, 4, 5}));
        EXPECT_EQ(grouped_space_to_depth->get_input_node_ptr(0)->get_type_name(), std::string("Parameter"));
    }
}

TEST(FuseAvgDownTest, RewritesDynamicTemporalPadsWithStaticSourceChannels) {
    const std::vector<std::pair<int64_t, int64_t>> channel_transitions = {
        {96, 192},
        {192, 384},
        {384, 768},
    };
    for (const auto& [input_channels, output_channels] : channel_transitions) {
        SCOPED_TRACE(input_channels);
        auto model = make_avg_down_model(input_channels, output_channels, 2, 2, true, true, false, true);
        const auto pad = find_pad(model);
        ASSERT_TRUE(pad);
        pad->set_output_type(0, pad->get_output_element_type(0), ov::PartialShape::dynamic(5));
        ASSERT_TRUE(pad->get_output_partial_shape(0)[1].is_dynamic());

        run_pass(model);

        EXPECT_TRUE(ov::is_type<ov::intel_gpu::op::GroupedSpaceToDepth>(model->get_results().front()->get_input_node_shared_ptr(0)));
    }

    auto identity_model = make_avg_down_model(768, 768, 1, 1, true, true, true, true);
    const auto identity_pad = find_pad(identity_model);
    ASSERT_TRUE(identity_pad);
    identity_pad->set_output_type(0, identity_pad->get_output_element_type(0), ov::PartialShape::dynamic(5));
    ASSERT_TRUE(identity_pad->get_output_partial_shape(0)[1].is_dynamic());

    run_pass(identity_model);

    const auto consumer = identity_model->get_results().front()->get_input_node_shared_ptr(0);
    ASSERT_EQ(consumer->get_type_name(), std::string("Relu"));
    EXPECT_EQ(consumer->get_input_node_ptr(0)->get_type_name(), std::string("Parameter"));
}

TEST(FuseAvgDownTest, RejectsMisplacedDynamicTemporalPadding) {
    auto model = make_avg_down_model(96, 192, 2, 2, true, true, false, true, DynamicPaddingVariant::MisplacedTemporal);

    run_pass(model);

    EXPECT_EQ(model->get_results().front()->get_input_node_ptr(0)->get_type_name(), std::string("ReduceMean"));
}

TEST(FuseAvgDownTest, RejectsNonzeroDynamicSpatialPadding) {
    auto model = make_avg_down_model(96, 192, 2, 2, true, true, false, true, DynamicPaddingVariant::NonzeroSpatial);

    run_pass(model);

    EXPECT_EQ(model->get_results().front()->get_input_node_ptr(0)->get_type_name(), std::string("ReduceMean"));
}

TEST(FuseAvgDownTest, RejectsUnexpectedGroupSize) {
    auto model = make_avg_down_model(96, 128, 2, 2, true);
    run_pass(model);

    EXPECT_EQ(model->get_results().front()->get_input_node_ptr(0)->get_type_name(), std::string("ReduceMean"));
}

TEST(GroupedSpaceToDepthTest, InfersIntervalDimensions) {
    auto parameter =
        std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                ov::PartialShape{ov::Dimension(1, 4), 96, ov::Dimension(3, 8), ov::Dimension(8, 12), ov::Dimension(10, 14)});
    auto grouped_space_to_depth = std::make_shared<ov::intel_gpu::op::GroupedSpaceToDepth>(parameter, 2, 2, 192);

    EXPECT_EQ(grouped_space_to_depth->get_output_partial_shape(0),
              ov::PartialShape({ov::Dimension(1, 4), 192, ov::Dimension(2, 4), ov::Dimension(4, 6), ov::Dimension(5, 7)}));
}

TEST(FuseAvgDownTest, RejectsUnexpectedTransposeOrder) {
    auto model = make_avg_down_model(96, 96, 1, 2, false, false);
    run_pass(model);

    EXPECT_EQ(model->get_results().front()->get_input_node_ptr(0)->get_type_name(), std::string("ReduceMean"));
}

}  // namespace
