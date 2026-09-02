// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/transformations/remove_fq_before_dw_conv.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <vector>

#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/mvn.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/manager.hpp"

namespace ov::test::intel_gpu {

namespace {

struct GraphOptions {
    bool shared_parent = true;
    bool shared_fake_quantize = false;
    bool with_mvn = true;
    bool mvn_reduces_channel = true;
    bool channelwise_fake_quantize = true;
    bool constant_fake_quantize_bounds = true;
    bool quantized_weights = true;
    bool dynamic_spatial_shape = false;
    size_t groups = 8;
    size_t input_channels_per_group = 1;
    size_t output_channels_per_group = 1;
    size_t kernel_size = 3;
};

struct TestGraph {
    std::shared_ptr<ov::Model> model;
    std::shared_ptr<ov::op::v0::Relu> parent;
    std::shared_ptr<ov::op::v0::FakeQuantize> fake_quantize;
    std::shared_ptr<ov::op::v1::GroupConvolution> convolution;
    std::shared_ptr<ov::Node> weights;
    std::shared_ptr<ov::op::v0::Relu> parent_consumer;
};

TestGraph make_test_graph(const GraphOptions& options) {
    const size_t in_channels = options.groups * options.input_channels_per_group;
    const size_t output_channels = options.groups * options.output_channels_per_group;
    const ov::PartialShape data_shape = options.dynamic_spatial_shape ? ov::PartialShape{1, ov::Dimension::value_type(in_channels), -1, -1} : ov::PartialShape{1, ov::Dimension::value_type(in_channels), 8, 8};

    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, data_shape);
    auto parent = std::make_shared<ov::op::v0::Relu>(input);
    ov::ParameterVector parameters{input};

    const ov::Shape fq_shape = options.channelwise_fake_quantize ? ov::Shape{1, in_channels, 1, 1} : ov::Shape{};
    auto make_fq_bound = [&](float value) -> ov::Output<ov::Node> {
        if (options.constant_fake_quantize_bounds) {
            return ov::op::v0::Constant::create(ov::element::f16, fq_shape, {value});
        }

        auto parameter = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, fq_shape);
        parameters.push_back(parameter);
        return parameter;
    };

    auto fake_quantize =
        std::make_shared<ov::op::v0::FakeQuantize>(parent, make_fq_bound(-1.f), make_fq_bound(1.f), make_fq_bound(-128.f), make_fq_bound(127.f), 256);

    const ov::Shape weights_shape{options.groups,
                                  options.output_channels_per_group,
                                  options.input_channels_per_group,
                                  options.kernel_size,
                                  options.kernel_size};
    std::shared_ptr<ov::Node> weights;
    if (options.quantized_weights) {
        auto compressed_weights = ov::op::v0::Constant::create(ov::element::i8, weights_shape, {1});
        auto convert = std::make_shared<ov::op::v0::Convert>(compressed_weights, ov::element::f16);
        auto scale = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{options.groups, options.output_channels_per_group, 1, 1, 1}, {0.1f});
        weights = std::make_shared<ov::op::v1::Multiply>(convert, scale);
    } else {
        weights = ov::op::v0::Constant::create(ov::element::f16, weights_shape, {0.1f});
    }

    const auto padding = static_cast<ptrdiff_t>(options.kernel_size / 2);
    auto convolution = std::make_shared<ov::op::v1::GroupConvolution>(fake_quantize,
                                                                      weights,
                                                                      ov::Strides{1, 1},
                                                                      ov::CoordinateDiff{padding, padding},
                                                                      ov::CoordinateDiff{padding, padding},
                                                                      ov::Strides{1, 1});
    auto bias = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1, output_channels, 1, 1}, {0.f});
    auto add = std::make_shared<ov::op::v1::Add>(convolution, bias);
    auto order = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 3, 1});
    auto transpose = std::make_shared<ov::op::v1::Transpose>(add, order);

    std::shared_ptr<ov::Node> main_output = transpose;
    if (options.with_mvn) {
        const std::vector<int64_t> axes_values = options.mvn_reduces_channel ? std::vector<int64_t>{-1} : std::vector<int64_t>{1};
        auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, axes_values);
        main_output = std::make_shared<ov::op::v6::MVN>(transpose, axes, true, 1e-6f, ov::op::MVNEpsMode::INSIDE_SQRT);
    }

    ov::OutputVector outputs{main_output};
    std::shared_ptr<ov::op::v0::Relu> parent_consumer;
    if (options.shared_parent) {
        parent_consumer = std::make_shared<ov::op::v0::Relu>(parent);
        outputs.push_back(parent_consumer);
    }
    if (options.shared_fake_quantize) {
        outputs.push_back(std::make_shared<ov::op::v0::Relu>(fake_quantize));
    }

    return {std::make_shared<ov::Model>(outputs, parameters), parent, fake_quantize, convolution, weights, parent_consumer};
}

void run_pass_and_check(const GraphOptions& options, bool expect_removal) {
    auto graph = make_test_graph(options);

    ov::pass::Manager manager;
    manager.register_pass<ov::intel_gpu::RemoveFakeQuantizeBeforeDepthwiseConv>();
    manager.run_passes(graph.model);

    EXPECT_EQ(graph.convolution->get_input_node_shared_ptr(1), graph.weights);
    if (expect_removal) {
        EXPECT_EQ(graph.convolution->get_input_node_shared_ptr(0), graph.parent);
        EXPECT_TRUE(graph.fake_quantize->output(0).get_target_inputs().empty());
        const auto ordered_ops = graph.model->get_ordered_ops();
        EXPECT_TRUE(std::none_of(ordered_ops.begin(), ordered_ops.end(), [](const std::shared_ptr<ov::Node>& node) {
            return ov::is_type<ov::op::v0::FakeQuantize>(node);
        }));
        ASSERT_NE(graph.parent_consumer, nullptr);
        EXPECT_EQ(graph.parent_consumer->get_input_node_shared_ptr(0), graph.parent);
    } else {
        EXPECT_EQ(graph.convolution->get_input_node_shared_ptr(0), graph.fake_quantize);
    }
}

}  // namespace

TEST(RemoveFakeQuantizeBeforeDepthwiseConvTest, RemovesMatchingFakeQuantize) {
    run_pass_and_check(GraphOptions{}, true);
}

TEST(RemoveFakeQuantizeBeforeDepthwiseConvTest, KeepsFakeQuantizeForSingleConsumerParent) {
    GraphOptions options;
    options.shared_parent = false;
    run_pass_and_check(options, false);
}

TEST(RemoveFakeQuantizeBeforeDepthwiseConvTest, KeepsSharedFakeQuantize) {
    GraphOptions options;
    options.shared_fake_quantize = true;
    run_pass_and_check(options, false);
}

TEST(RemoveFakeQuantizeBeforeDepthwiseConvTest, KeepsFakeQuantizeForNonDepthwiseConvolution) {
    GraphOptions options;
    options.groups = 4;
    options.input_channels_per_group = 2;
    options.output_channels_per_group = 2;
    run_pass_and_check(options, false);
}

TEST(RemoveFakeQuantizeBeforeDepthwiseConvTest, KeepsFakeQuantizeForChannelMultiplier) {
    GraphOptions options;
    options.output_channels_per_group = 2;
    run_pass_and_check(options, false);
}

TEST(RemoveFakeQuantizeBeforeDepthwiseConvTest, KeepsFakeQuantizeForLargeKernel) {
    GraphOptions options;
    options.kernel_size = 5;
    run_pass_and_check(options, false);
}

TEST(RemoveFakeQuantizeBeforeDepthwiseConvTest, KeepsFakeQuantizeWithoutMvn) {
    GraphOptions options;
    options.with_mvn = false;
    run_pass_and_check(options, false);
}

TEST(RemoveFakeQuantizeBeforeDepthwiseConvTest, KeepsFakeQuantizeWhenMvnDoesNotReduceChannel) {
    GraphOptions options;
    options.mvn_reduces_channel = false;
    run_pass_and_check(options, false);
}

TEST(RemoveFakeQuantizeBeforeDepthwiseConvTest, KeepsPerTensorFakeQuantize) {
    GraphOptions options;
    options.channelwise_fake_quantize = false;
    run_pass_and_check(options, false);
}

TEST(RemoveFakeQuantizeBeforeDepthwiseConvTest, KeepsFakeQuantizeWithDynamicBounds) {
    GraphOptions options;
    options.constant_fake_quantize_bounds = false;
    run_pass_and_check(options, false);
}

TEST(RemoveFakeQuantizeBeforeDepthwiseConvTest, KeepsFakeQuantizeWithFloatingPointWeights) {
    GraphOptions options;
    options.quantized_weights = false;
    run_pass_and_check(options, false);
}

TEST(RemoveFakeQuantizeBeforeDepthwiseConvTest, KeepsFakeQuantizeForDynamicShape) {
    GraphOptions options;
    options.dynamic_spatial_shape = true;
    run_pass_and_check(options, true);
}

}  // namespace ov::test::intel_gpu
