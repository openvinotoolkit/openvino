// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "lazy_tensor.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "ov_ops/moe_compressed.hpp"
#include "partitioning/patterns/opt.hpp"

namespace {

using ov::op::internal::MOE;
using ov::op::internal::MOECompressed;
using ov::npuw::patterns::opt::Context;
using ov::npuw::patterns::opt::MoECompressedScaleGroupMajor;

constexpr size_t kExperts = 4;
constexpr size_t kOutRows = 64;
constexpr size_t kGroupSize = 16;
constexpr size_t kTokens = 2;
constexpr size_t kTopK = 2;
const std::vector<size_t> kGroupMajor = {0, 2, 1};

struct Options {
    bool has_zp = false;
    size_t groups = 4;
    ov::element::Type scale_type = ov::element::f16;
    bool rank4_scales = false;
    bool share_gate_up_scale = false;
    // Number of leading scales (gate, up, down) which are function inputs instead of closures
    size_t non_closure_scales = 0;
};

struct Graph {
    std::shared_ptr<ov::Model> model;
    std::shared_ptr<MOECompressed> moe;
    ov::ParameterVector scales;  // one per scale input of the MOE (gate, up, down)
    size_t param_base_idx = 0;
};

constexpr size_t kScaleInputs[] = {4, 7, 10};

Graph make_graph(const Options& opts = {}) {
    using ov::op::v0::Constant;
    using ov::op::v0::Parameter;

    const size_t K = opts.groups * kGroupSize;

    auto hidden_states = std::make_shared<Parameter>(ov::element::f16, ov::Shape{kTokens, K});
    auto routing_weights = std::make_shared<Parameter>(ov::element::f16, ov::Shape{kTokens, kTopK});
    auto topk_idx = std::make_shared<Parameter>(ov::element::i32, ov::Shape{kTokens, kTopK});

    ov::ParameterVector params{hidden_states, routing_weights, topk_idx};

    auto make_weight = [&] {
        return Constant::create(ov::element::u4, ov::Shape{kExperts, kOutRows, opts.groups, kGroupSize}, {1});
    };
    auto make_scale = [&] {
        ov::Shape shape{kExperts, kOutRows, opts.groups};
        if (opts.rank4_scales) {
            shape.push_back(1);
        }
        return std::make_shared<Parameter>(opts.scale_type, shape);
    };
    auto make_zp = [&]() -> ov::Output<ov::Node> {
        if (opts.has_zp) {
            return Constant::create(ov::element::u4, ov::Shape{kExperts, kOutRows, opts.groups, 1}, {0})->output(0);
        }
        // Absent zero point: a dynamic-typed placeholder
        auto zp = std::make_shared<Parameter>(ov::element::dynamic, ov::Shape{0});
        params.push_back(zp);
        return zp->output(0);
    };

    ov::OutputVector args{hidden_states, routing_weights, topk_idx};
    std::vector<std::shared_ptr<Parameter>> scales;
    for (size_t i = 0; i < 3; ++i) {
        std::shared_ptr<Parameter> scale;
        if (i == 1 && opts.share_gate_up_scale) {
            scale = scales.front();
        } else {
            scale = make_scale();
            scale->set_friendly_name(std::string("scale_") + std::to_string(i));
        }
        scales.push_back(scale);
        args.push_back(make_weight());
        args.push_back(scale);
        args.push_back(make_zp());
    }

    // Function inputs come first, closures after them
    ov::ParameterVector scale_params;
    for (const auto& s : scales) {
        if (std::find(scale_params.begin(), scale_params.end(), s) == scale_params.end()) {
            scale_params.push_back(s);
        }
    }
    for (size_t i = 0; i < opts.non_closure_scales; ++i) {
        params.push_back(scale_params[i]);
    }
    const size_t base = params.size();
    for (size_t i = opts.non_closure_scales; i < scale_params.size(); ++i) {
        params.push_back(scale_params[i]);
    }

    MOECompressed::Config config;
    config.expert_type = MOE::Expert_type::GEMM3_SWIGLU;
    config.activation_type = MOE::Activation_type::SWIGLU;
    config.hidden_size = K;
    config.inter_size = K;
    config.num_expert = kExperts;
    config.top_k = kTopK;
    config.group_size = opts.groups == 1 ? std::numeric_limits<size_t>::max() : kGroupSize;
    config.has_zp = opts.has_zp;
    config.out_type = ov::element::dynamic;

    auto moe = std::make_shared<MOECompressed>(args, config);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(moe->output(0))},
                                             params,
                                             "moe_compressed_scale_gm");
    return {model, moe, ov::ParameterVector(scales.begin(), scales.end()), base};
}

void run_pass(Graph& graph, Context& ctx, bool to_f32 = false) {
    ov::pass::GraphRewrite rewr;
    rewr.add_matcher<MoECompressedScaleGroupMajor>(std::ref(ctx), to_f32);
    rewr.run_on_model(graph.model);
}

void expect_untouched(const Graph& graph, const Context& ctx, const ov::Shape& scale_shape) {
    EXPECT_TRUE(ctx.closures_to_permute.empty());
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(graph.moe->get_input_node_shared_ptr(kScaleInputs[i]), graph.scales[i]);
        EXPECT_EQ(graph.scales[i]->get_shape(), scale_shape);
    }
    size_t transposes = 0;
    for (const auto& node : graph.model->get_ordered_ops()) {
        transposes += ov::is_type<ov::op::v1::Transpose>(node) ? 1u : 0u;
    }
    EXPECT_EQ(transposes, 0u);
}

}  // namespace

TEST(MoECompressedScaleGroupMajorTest, RelayoutsAllThreeScales) {
    auto graph = make_graph();
    const ov::Shape original{kExperts, kOutRows, 4};
    const ov::Shape relaid{kExperts, 4, kOutRows};

    Context ctx(graph.model, graph.param_base_idx);
    run_pass(graph, ctx);
    ASSERT_NO_THROW(graph.model->validate_nodes_and_infer_types());

    ASSERT_EQ(ctx.closures_to_permute.size(), 3u);
    for (size_t i = 0; i < 3; ++i) {
        const auto& scale = graph.scales[i];
        // The closure is relaid out on host...
        ASSERT_EQ(ctx.closures_to_permute.count(scale), 1u);
        EXPECT_EQ(ctx.closures_to_permute.at(scale), kGroupMajor);
        EXPECT_EQ(scale->get_shape(), relaid);
        EXPECT_EQ(scale->get_element_type(), ov::element::f16);

        // ...and a Transpose restores the layout the MOE expects
        auto transpose = ov::as_type_ptr<ov::op::v1::Transpose>(graph.moe->get_input_node_shared_ptr(kScaleInputs[i]));
        ASSERT_NE(transpose, nullptr);
        EXPECT_EQ(transpose->get_input_node_shared_ptr(0), scale);
        auto order = ov::as_type_ptr<ov::op::v0::Constant>(transpose->get_input_node_shared_ptr(1));
        ASSERT_NE(order, nullptr);
        EXPECT_EQ(order->cast_vector<int64_t>(), (std::vector<int64_t>{0, 2, 1}));
        EXPECT_EQ(transpose->get_output_shape(0), original);
    }

    // Weights and the rest of the inputs are not touched
    for (size_t idx : {3u, 6u, 9u}) {
        EXPECT_TRUE(ov::is_type<ov::op::v0::Constant>(graph.moe->get_input_node_shared_ptr(idx)));
    }
    EXPECT_EQ(graph.moe->get_output_shape(0), (ov::Shape{kTokens, 4 * kGroupSize}));
}

TEST(MoECompressedScaleGroupMajorTest, RelayoutsF32Scales) {
    Options opts;
    opts.scale_type = ov::element::f32;
    auto graph = make_graph(opts);

    Context ctx(graph.model, graph.param_base_idx);
    run_pass(graph, ctx);

    EXPECT_EQ(ctx.closures_to_permute.size(), 3u);
    for (const auto& scale : graph.scales) {
        EXPECT_EQ(scale->get_element_type(), ov::element::f32);
        EXPECT_EQ(scale->get_shape(), (ov::Shape{kExperts, 4, kOutRows}));
    }
}

TEST(MoECompressedScaleGroupMajorTest, ConvertsF16ScalesToF32WhenRequested) {
    auto graph = make_graph();

    Context ctx(graph.model, graph.param_base_idx);
    run_pass(graph, ctx, /*to_f32=*/true);
    ASSERT_NO_THROW(graph.model->validate_nodes_and_infer_types());

    EXPECT_EQ(ctx.closures_to_permute.size(), 3u);
    EXPECT_EQ(ctx.closures_to_f32.size(), 3u);
    for (size_t i = 0; i < 3; ++i) {
        const auto& scale = graph.scales[i];
        EXPECT_EQ(ctx.closures_to_f32.count(scale), 1u);
        EXPECT_EQ(scale->get_element_type(), ov::element::f32);
        EXPECT_EQ(scale->get_shape(), (ov::Shape{kExperts, 4, kOutRows}));
        // Parameter -> Transpose -> MOECompressed, nothing in between
        auto transpose = ov::as_type_ptr<ov::op::v1::Transpose>(graph.moe->get_input_node_shared_ptr(kScaleInputs[i]));
        ASSERT_NE(transpose, nullptr);
        EXPECT_EQ(transpose->get_input_node_shared_ptr(0), scale);
        EXPECT_EQ(graph.moe->get_input_element_type(kScaleInputs[i]), ov::element::f32);
    }
}

TEST(MoECompressedScaleGroupMajorTest, F32ScalesNeedNoConversion) {
    Options opts;
    opts.scale_type = ov::element::f32;
    auto graph = make_graph(opts);

    Context ctx(graph.model, graph.param_base_idx);
    run_pass(graph, ctx, /*to_f32=*/true);

    EXPECT_EQ(ctx.closures_to_permute.size(), 3u);
    EXPECT_TRUE(ctx.closures_to_f32.empty());
}

TEST(MoECompressedScaleGroupMajorTest, NoConversionByDefault) {
    auto graph = make_graph();

    Context ctx(graph.model, graph.param_base_idx);
    run_pass(graph, ctx);

    EXPECT_TRUE(ctx.closures_to_f32.empty());
    for (const auto& scale : graph.scales) {
        EXPECT_EQ(scale->get_element_type(), ov::element::f16);
    }
}

// The partitioner builds permute(...).convert(f32) for these closures
TEST(MoECompressedScaleGroupMajorTest, F32ClosureDataIsPermutedAndConverted) {
    constexpr size_t G = 4;
    std::vector<ov::float16> raw(kExperts * kOutRows * G);
    for (size_t i = 0; i < raw.size(); ++i) {
        raw[i] = ov::float16(static_cast<float>(i % 1024) * 0.125f);
    }
    auto source = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{kExperts, kOutRows, G}, raw.data());

    auto graph = make_graph();
    Context ctx(graph.model, graph.param_base_idx);
    run_pass(graph, ctx, /*to_f32=*/true);
    ASSERT_EQ(ctx.closures_to_f32.count(graph.scales[0]), 1u);

    auto lazy = ov::npuw::weights::LazyTensor(source)
                    .permute(ctx.closures_to_permute.at(graph.scales[0]))
                    .convert(ov::element::f32);
    const auto meta = lazy.eval_meta();
    EXPECT_EQ(meta.shape, graph.scales[0]->get_shape());
    EXPECT_EQ(meta.type, graph.scales[0]->get_element_type());

    const auto evaluated = lazy.eval();
    ASSERT_EQ(evaluated.get_element_type(), ov::element::f32);
    ASSERT_EQ(evaluated.get_shape(), (ov::Shape{kExperts, G, kOutRows}));
    const auto* data = static_cast<const float*>(evaluated.data());
    for (size_t e = 0; e < kExperts; ++e) {
        for (size_t g = 0; g < G; ++g) {
            for (size_t n = 0; n < kOutRows; ++n) {
                ASSERT_EQ(data[(e * G + g) * kOutRows + n], static_cast<float>(raw[(e * kOutRows + n) * G + g]))
                    << "e=" << e << " g=" << g << " n=" << n;
            }
        }
    }
}

// The lazy closure which the partitioner builds from ctx.closures_to_permute holds the group-major data
TEST(MoECompressedScaleGroupMajorTest, ClosureDataIsPermutedOriginal) {
    constexpr size_t G = 4;
    std::vector<uint16_t> raw(kExperts * kOutRows * G);
    std::iota(raw.begin(), raw.end(), uint16_t{1});
    auto source = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{kExperts, kOutRows, G}, raw.data());

    auto graph = make_graph();
    Context ctx(graph.model, graph.param_base_idx);
    run_pass(graph, ctx);
    ASSERT_EQ(ctx.closures_to_permute.size(), 3u);

    auto lazy = ov::npuw::weights::LazyTensor(source).permute(ctx.closures_to_permute.at(graph.scales[0]));
    const auto meta = lazy.eval_meta();
    EXPECT_EQ(meta.shape, graph.scales[0]->get_shape());
    EXPECT_EQ(meta.type, graph.scales[0]->get_element_type());

    const auto evaluated = lazy.eval();
    ASSERT_EQ(evaluated.get_shape(), (ov::Shape{kExperts, G, kOutRows}));
    const auto* data = static_cast<const uint16_t*>(evaluated.data());
    for (size_t e = 0; e < kExperts; ++e) {
        for (size_t g = 0; g < G; ++g) {
            for (size_t n = 0; n < kOutRows; ++n) {
                ASSERT_EQ(data[(e * G + g) * kOutRows + n], raw[(e * kOutRows + n) * G + g])
                    << "e=" << e << " g=" << g << " n=" << n;
            }
        }
    }
}

TEST(MoECompressedScaleGroupMajorTest, SkipsWhenZeroPointsArePresent) {
    Options opts;
    opts.has_zp = true;
    auto graph = make_graph(opts);
    Context ctx(graph.model, graph.param_base_idx);
    run_pass(graph, ctx);
    expect_untouched(graph, ctx, ov::Shape{kExperts, kOutRows, 4});
}

TEST(MoECompressedScaleGroupMajorTest, SkipsPerChannelScales) {
    Options opts;
    opts.groups = 1;
    auto graph = make_graph(opts);
    Context ctx(graph.model, graph.param_base_idx);
    run_pass(graph, ctx);
    expect_untouched(graph, ctx, ov::Shape{kExperts, kOutRows, 1});
}

TEST(MoECompressedScaleGroupMajorTest, SkipsWhenAnyScaleIsNotAClosure) {
    Options opts;
    opts.non_closure_scales = 1;
    auto graph = make_graph(opts);
    Context ctx(graph.model, graph.param_base_idx);
    run_pass(graph, ctx);
    // All or none: the two closure scales stay as they are too
    expect_untouched(graph, ctx, ov::Shape{kExperts, kOutRows, 4});
}

TEST(MoECompressedScaleGroupMajorTest, SkipsRank4Scales) {
    Options opts;
    opts.rank4_scales = true;
    auto graph = make_graph(opts);
    Context ctx(graph.model, graph.param_base_idx);
    run_pass(graph, ctx);
    expect_untouched(graph, ctx, ov::Shape{kExperts, kOutRows, 4, 1});
}

TEST(MoECompressedScaleGroupMajorTest, SkipsWhenAScaleIsSharedBetweenInputs) {
    Options opts;
    opts.share_gate_up_scale = true;
    auto graph = make_graph(opts);
    Context ctx(graph.model, graph.param_base_idx);
    run_pass(graph, ctx);
    EXPECT_TRUE(ctx.closures_to_permute.empty());
    EXPECT_EQ(graph.moe->get_input_node_shared_ptr(4), graph.scales[0]);
    EXPECT_EQ(graph.moe->get_input_node_shared_ptr(7), graph.scales[0]);
}

TEST(MoECompressedScaleGroupMajorTest, SecondRunLeavesRelaidGraphAlone) {
    auto graph = make_graph();
    Context ctx(graph.model, graph.param_base_idx);
    run_pass(graph, ctx);
    ASSERT_EQ(ctx.closures_to_permute.size(), 3u);

    Context again(graph.model, graph.param_base_idx);
    run_pass(graph, again);
    EXPECT_TRUE(again.closures_to_permute.empty());
}
