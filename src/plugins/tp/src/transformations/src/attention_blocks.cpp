// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <unordered_set>

#include "block_utils.hpp"
#include "openvino/core/except.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "tp_transformations/analyzer.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {
namespace tp {

namespace {

using namespace blocks;

constexpr const char* kQkvNames[3] = {"query", "key", "value"};

/// Nodes reached walking back from `output` through transparent ops, the first opaque ones included.
std::vector<std::shared_ptr<ov::Node>> producers_through_transparent(const ov::Output<ov::Node>& output) {
    std::vector<std::shared_ptr<ov::Node>> reached;
    std::unordered_set<ov::Node*> visited;
    ov::op::util::visit_path(
        output.get_node(),
        visited,
        [&](ov::Node* node) {
            reached.push_back(node->shared_from_this());
        },
        [&](ov::Node* node) {
            // visit_path does not report the nodes it stops at.
            const bool stop = ov::is_type<ov::op::v0::MatMul>(node) || !is_transparent(node);
            if (stop)
                reached.push_back(node->shared_from_this());
            return stop;
        });
    return reached;
}

/// The Reshape that flattens the heads back into features in front of the O projection.
std::shared_ptr<ov::Node> output_reshape(const std::shared_ptr<ov::Node>& out_projection) {
    auto node = out_projection->get_input_node_shared_ptr(0);
    for (int hops = 0; hops < 4 && node && !ov::is_type<ov::op::v1::Reshape>(node); ++hops) {
        node = ov::is_type<ov::op::v0::Convert>(node) ? node->get_input_node_shared_ptr(0) : nullptr;
    }
    return node && ov::is_type<ov::op::v1::Reshape>(node) ? node : nullptr;
}

struct Geometry {
    int64_t heads = 0;
    int64_t kv_heads = 0;
    int64_t head_dim = 0;
};

/// The parts every attention block shares: projections, their head Reshapes and the O-side Reshape.
/// Throws when the block is only partly recognized.
Geometry annotate_attention(const std::shared_ptr<ov::Node>& attention,
                            const TPAnalyzerConfig& config,
                            TPAnalysisReport& report,
                            const std::string& block) {
    std::shared_ptr<ov::Node> qkv[3];
    for (size_t i = 0; i < 3; ++i) {
        OPENVINO_ASSERT(i < attention->get_input_size(),
                        "[TP] Attention node '", attention->get_friendly_name(), "' has no ", kQkvNames[i], " input");
        const auto producers = nearest_producer_matmuls(attention->input_value(i));
        OPENVINO_ASSERT(producers.size() == 1,
                        "[TP] Expected exactly one projection feeding the ", kQkvNames[i], " input of '",
                        attention->get_friendly_name(), "', found ", producers.size(),
                        ". A projection is a MatMul with a constant weight; none is found when the weight is "
                        "produced at runtime (a LoRA adapter, for instance). Model may not be supported.");
        qkv[i] = producers.front();
    }
    const auto out = nearest_consumer_matmul(attention);
    OPENVINO_ASSERT(out != nullptr,
                    "[TP] Could not find the out projection following '", attention->get_friendly_name(),
                    "'. Model may not be supported.");

    Geometry g;
    OPENVINO_ASSERT(read_head_layout(qkv[0], g.heads, g.head_dim),
                    "[TP] Could not determine the head layout of '", attention->get_friendly_name(),
                    "': its query projection is not followed by a Reshape with a static shape constant");
    const int64_t q_features = weight_dim(qkv[0], weight_axes(qkv[0]).out_features);
    const int64_t kv_features = weight_dim(qkv[1], weight_axes(qkv[1]).out_features);
    OPENVINO_ASSERT(g.heads > 0 && g.head_dim > 0 && kv_features > 0 && kv_features % g.head_dim == 0,
                    "[TP] '", attention->get_friendly_name(), "': ", g.heads, " heads of ", g.head_dim,
                    " and a key projection of ", kv_features, " features do not form a head layout");
    g.kv_heads = kv_features / g.head_dim;
    OPENVINO_ASSERT(g.heads % g.kv_heads == 0,
                    "[TP] ", g.heads, " query heads do not group evenly over ", g.kv_heads, " KV heads");
    OPENVINO_ASSERT(q_features == g.heads * g.head_dim,
                    "[TP] The query projection produces ", q_features, " features, which does not match ", g.heads,
                    " heads x ", g.head_dim, " head_dim");
    OPENVINO_ASSERT(static_cast<int64_t>(config.world_size) <= g.kv_heads,
                    "[TP] The model has ", g.kv_heads, " KV heads, so it can be split across at most that many ranks; ",
                    config.world_size, " devices were requested. Replicating KV heads across ranks is not supported yet.");

    // Query heads follow their KV head, so every rank owns whole query groups.
    const auto group = static_cast<size_t>(g.heads / g.kv_heads);
    const auto head_dim = static_cast<size_t>(g.head_dim);
    const std::vector<Segment> q_features_split{{static_cast<size_t>(q_features), group * head_dim}};
    const std::vector<Segment> kv_features_split{{static_cast<size_t>(kv_features), head_dim}};
    const std::vector<Segment> q_heads{{static_cast<size_t>(g.heads), group}};
    const std::vector<Segment> kv_heads{{static_cast<size_t>(g.kv_heads), 1}};

    auto project = [&](const std::shared_ptr<ov::Node>& matmul,
                       ShardStrategy strategy,
                       const std::vector<Segment>& segments,
                       Collective collective) {
        if (!annotate_projection(matmul, strategy, segments, collective, config.world_size, block))
            report.runtime_slices.push_back(matmul->get_friendly_name());
    };
    project(qkv[0], ShardStrategy::COLUMN_PARALLEL, q_features_split, Collective::NONE);
    project(qkv[1], ShardStrategy::COLUMN_PARALLEL, kv_features_split, Collective::NONE);
    project(qkv[2], ShardStrategy::COLUMN_PARALLEL, kv_features_split, Collective::NONE);
    project(out, ShardStrategy::ROW_PARALLEL, q_features_split, Collective::ALL_REDUCE);

    // [.., features] -> [.., heads, head_dim] after every projection.
    for (size_t i = 0; i < 3; ++i) {
        for (const auto& target : projection_output(qkv[i]).get_target_inputs()) {
            const auto reshape = target.get_node()->shared_from_this();
            if (!ov::is_type<ov::op::v1::Reshape>(reshape))
                continue;
            const auto shape = ov::as_type_ptr<ov::op::v0::Constant>(reshape->get_input_node_shared_ptr(1));
            if (shape && shape_size(shape->get_shape()) > 2)
                annotate_shape_input(reshape, 1, {2}, i == 0 ? q_heads : kv_heads, block);
        }
    }

    // [.., heads, head_dim] -> [.., features] in front of the O projection; exports that compute
    // this shape with ShapeOf + Concat are relative already.
    if (const auto reshape = output_reshape(out)) {
        if (const auto shape = ov::as_type_ptr<ov::op::v0::Constant>(reshape->get_input_node_shared_ptr(1))) {
            std::vector<size_t> elements;
            const auto data = shape->cast_vector<int64_t>();
            for (size_t e = 0; e < data.size(); ++e) {
                if (data[e] == q_features)
                    elements.push_back(e);
            }
            if (!elements.empty())
                annotate_shape_input(reshape, 1, elements, q_features_split, block);
        }
    }

    ++report.attention_blocks;
    if (report.num_heads == 0) {
        report.num_heads = g.heads;
        report.num_kv_heads = g.kv_heads;
        report.head_dim = g.head_dim;
    }
    return g;
}

}  // namespace

TPPagedAttentionBlock::TPPagedAttentionBlock(const TPAnalyzerConfig& config, TPAnalysisReportPtr report) {
    auto pa = ov::pass::pattern::wrap_type<ov::op::PagedAttentionExtension>();
    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(pa, get_type_info().name),
                     [=](ov::pass::pattern::Matcher& m) {
                         const auto attention = m.get_match_root();
                         const auto g = annotate_attention(attention, config, *report, "TPPagedAttentionBlock");

                         // The conversion records the KV head count in rt_info, where the cache sizing
                         // and the device plugin read it.
                         ShardAnnotation heads(ShardStrategy::COLUMN_PARALLEL, -1, {{static_cast<size_t>(g.kv_heads), 1}},
                                               "TPPagedAttentionBlock");
                         for (const auto* key : {"num_k_heads", "num_v_heads"}) {
                             const auto it = attention->get_rt_info().find(key);
                             if (it == attention->get_rt_info().end())
                                 continue;
                             OPENVINO_ASSERT(it->second.as<int64_t>() == g.kv_heads,
                                             "[TP] '", attention->get_friendly_name(), "' declares ",
                                             it->second.as<int64_t>(), " for '", key, "' where the model has ",
                                             g.kv_heads, " kv heads");
                             heads.rt_keys.emplace_back(key);
                         }
                         if (!heads.rt_keys.empty())
                             set_shard(attention, heads);
                         return false;
                     });
}

TPStatefulSDPABlock::TPStatefulSDPABlock(const TPAnalyzerConfig& config, TPAnalysisReportPtr report) {
    auto sdpa = ov::pass::pattern::wrap_type<ov::op::v13::ScaledDotProductAttention>();
    register_matcher(
        std::make_shared<ov::pass::pattern::Matcher>(sdpa, get_type_info().name),
        [=](ov::pass::pattern::Matcher& m) {
            const auto attention = m.get_match_root();
            const std::string block = "TPStatefulSDPABlock";
            const auto g = annotate_attention(attention, config, *report, block);
            const std::vector<Segment> q_heads{{static_cast<size_t>(g.heads), static_cast<size_t>(g.heads / g.kv_heads)}};
            const std::vector<Segment> kv_heads{{static_cast<size_t>(g.kv_heads), 1}};

            auto kv_bcst = ov::op::util::match_multi_query_bcst(ov::pass::pattern::any_input());
            ov::pass::pattern::Matcher broadcast(std::get<0>(kv_bcst), "TPMultiQueryBcst");
            size_t broadcasts = 0;

            for (size_t i = 1; i < 3; ++i) {
                for (const auto& node : producers_through_transparent(attention->input_value(i))) {
                    // The GQA broadcast from KV heads up to query heads, merged back into
                    // [B, heads, S, head_dim].
                    if (ov::is_type<ov::op::v1::Reshape>(node) && g.kv_heads != g.heads &&
                        broadcast.match(node->output(0))) {
                        const auto shape = ov::as_type_ptr<ov::op::v0::Constant>(node->get_input_node_shared_ptr(1));
                        if (shape) {
                            const auto data = shape->cast_vector<int64_t>();
                            if (data.size() == 4 && data[1] == g.heads) {
                                annotate_shape_input(node, 1, {1}, q_heads, block);
                                ++broadcasts;
                            }
                        }
                    }

                    // The KV cache: ReadValue -> [Gather(beam_idx)] -> Concat(past, new) -> Assign.
                    const auto read = ov::as_type_ptr<ov::op::v6::ReadValue>(node);
                    if (!read)
                        continue;
                    const auto& shape = read->get_variable()->get_info().data_shape;
                    if (!shape.rank().is_static() || shape.rank().get_length() != 4 || !shape[1].is_static() ||
                        shape[1].get_length() != g.kv_heads)
                        continue;
                    set_shard(read, ShardAnnotation(ShardStrategy::COLUMN_PARALLEL, 1, kv_heads, block));
                    const auto& id = read->get_variable()->get_info().variable_id;
                    if (std::find(report->sharded_variables.begin(), report->sharded_variables.end(), id) ==
                        report->sharded_variables.end())
                        report->sharded_variables.push_back(id);

                    // Its initializer: Broadcast(0, Concat(.., Constant{kv_heads}, ..)).
                    if (read->get_input_size() == 0)
                        continue;
                    const auto init = read->get_input_node_shared_ptr(0);
                    if (!ov::is_type_any_of<ov::op::v1::Broadcast, ov::op::v3::Broadcast>(init))
                        continue;
                    const auto concat = init->get_input_node_shared_ptr(1);
                    if (!ov::is_type<ov::op::v0::Concat>(concat))
                        continue;
                    for (size_t port = 0; port < concat->get_input_size(); ++port) {
                        const auto c = ov::as_type_ptr<ov::op::v0::Constant>(concat->get_input_node_shared_ptr(port));
                        if (c && c->get_shape() == ov::Shape{1} && c->cast_vector<int64_t>()[0] == g.kv_heads)
                            annotate_shape_input(concat, port, {0}, kv_heads, block);
                    }
                }
            }
            OPENVINO_ASSERT(g.kv_heads == g.heads || broadcasts > 0,
                            "[TP] The model uses grouped-query attention (", g.kv_heads, " KV heads for ", g.heads,
                            " Q heads) but no KV broadcast was found to re-shape at '",
                            attention->get_friendly_name(), "'");
            return false;
        });
}

}  // namespace tp
}  // namespace ov
