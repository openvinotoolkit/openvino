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
    /// Ranks sharing one KV head: world_size / kv_heads when there are fewer KV heads than ranks.
    size_t replicas = 1;
    std::vector<Segment> q_heads;
    std::vector<Segment> kv_heads_split;
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

    // A fused qkv_proj: one MatMul whose output is split into Q, K and V.
    const bool fused = qkv[0] == qkv[1] && qkv[1] == qkv[2];
    ov::Output<ov::Node> parts[3];
    if (fused) {
        const auto split = feature_parts(projection_output(qkv[0]));
        OPENVINO_ASSERT(split.size() == 3,
                        "[TP] The fused projection '", qkv[0]->get_friendly_name(), "' feeding '",
                        attention->get_friendly_name(), "' is not split into Q, K and V along its features");
        std::copy(split.begin(), split.end(), parts);
    } else {
        OPENVINO_ASSERT(qkv[0] != qkv[1] && qkv[1] != qkv[2] && qkv[0] != qkv[2],
                        "[TP] '", attention->get_friendly_name(), "' takes two of its inputs from one projection");
        for (size_t i = 0; i < 3; ++i)
            parts[i] = projection_output(qkv[i]);
    }
    auto features = [](const ov::Output<ov::Node>& part) -> int64_t {
        const auto& shape = part.get_partial_shape();
        const auto& dim = shape[shape.rank().get_length() - 1];
        return dim.is_static() ? dim.get_length() : 0;
    };

    Geometry g;
    OPENVINO_ASSERT(read_head_layout(parts[0], g.heads, g.head_dim),
                    "[TP] Could not determine the head layout of '", attention->get_friendly_name(),
                    "': its query projection is not followed by a Reshape with a static shape constant");
    const int64_t q_features = features(parts[0]);
    const int64_t kv_features = features(parts[1]);
    OPENVINO_ASSERT(features(parts[2]) == kv_features,
                    "[TP] '", attention->get_friendly_name(), "': the key and value projections differ in size");
    OPENVINO_ASSERT(g.heads > 0 && g.head_dim > 0 && kv_features > 0 && kv_features % g.head_dim == 0,
                    "[TP] '", attention->get_friendly_name(), "': ", g.heads, " heads of ", g.head_dim,
                    " and a key projection of ", kv_features, " features do not form a head layout");
    g.kv_heads = kv_features / g.head_dim;
    OPENVINO_ASSERT(g.heads % g.kv_heads == 0,
                    "[TP] ", g.heads, " query heads do not group evenly over ", g.kv_heads, " KV heads");
    OPENVINO_ASSERT(q_features == g.heads * g.head_dim,
                    "[TP] The query projection produces ", q_features, " features, which does not match ", g.heads,
                    " heads x ", g.head_dim, " head_dim");
    const auto world_size = static_cast<int64_t>(config.world_size);
    const bool even = world_size > g.kv_heads ? world_size % g.kv_heads == 0 && g.heads % world_size == 0
                                              : g.kv_heads % world_size == 0;
    OPENVINO_ASSERT(even,
                    "[TP] Model has ", g.heads, " Q heads and ", g.kv_heads, " KV heads; world size ", world_size,
                    " does not divide them evenly (required: KV heads % N == 0, or with fewer KV heads than ranks "
                    "N % KV heads == 0 and Q heads % N == 0). Every rank takes an equal share of the attention heads.");
    if (world_size > g.kv_heads)
        g.replicas = static_cast<size_t>(world_size / g.kv_heads);

    // Every rank owns whole query groups, or with fewer KV heads than ranks an equal share of one
    // group, next to the KV head(s) that group uses.
    const auto group = static_cast<size_t>(g.heads / g.kv_heads);
    const auto q_unit = group / g.replicas;
    const auto head_dim = static_cast<size_t>(g.head_dim);
    const size_t kv_count = g.replicas > 1 ? config.world_size / g.replicas : 0;
    const std::vector<Segment> q_features_split{{static_cast<size_t>(q_features), q_unit * head_dim}};
    const std::vector<Segment> kv_features_split{{static_cast<size_t>(kv_features), head_dim, g.replicas, kv_count}};
    g.q_heads = {{static_cast<size_t>(g.heads), q_unit}};
    g.kv_heads_split = {{static_cast<size_t>(g.kv_heads), 1, g.replicas, kv_count}};
    const auto& q_heads = g.q_heads;
    const auto& kv_heads = g.kv_heads_split;

    auto project = [&](const std::shared_ptr<ov::Node>& matmul,
                       ShardStrategy strategy,
                       const std::vector<Segment>& segments,
                       Collective collective) {
        if (!annotate_projection(matmul, strategy, segments, collective, config.world_size, block))
            report.runtime_slices.push_back(matmul->get_friendly_name());
    };
    project(out, ShardStrategy::ROW_PARALLEL, q_features_split, Collective::ALL_REDUCE);
    if (fused) {
        // Every rank's weight is its Q heads, then its K head(s), then its V head(s).
        const std::vector<Segment> qkv_split{q_features_split[0], kv_features_split[0], kv_features_split[0]};
        project(qkv[0], ShardStrategy::COLUMN_PARALLEL, qkv_split, Collective::NONE);
        annotate_feature_parts(projection_output(qkv[0]), qkv_split, block);
    } else {
        project(qkv[0], ShardStrategy::COLUMN_PARALLEL, q_features_split, Collective::NONE);
        project(qkv[1], ShardStrategy::COLUMN_PARALLEL, kv_features_split, Collective::NONE);
        project(qkv[2], ShardStrategy::COLUMN_PARALLEL, kv_features_split, Collective::NONE);
    }

    // [.., features] -> [.., heads, head_dim] after every projection.
    for (size_t i = 0; i < 3; ++i) {
        for (const auto& target : parts[i].get_target_inputs()) {
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
        report.kv_replicas = g.replicas;
    }
    return g;
}

/// Annotates element `element` of the shape vector feeding `consumer`'s input `port`, following Concats.
void annotate_size(const std::shared_ptr<ov::Node>& consumer,
                   size_t port,
                   size_t element,
                   int64_t expected,
                   const std::vector<Segment>& segments,
                   const std::string& block) {
    const auto source = consumer->get_input_node_shared_ptr(port);
    if (const auto c = ov::as_type_ptr<ov::op::v0::Constant>(source)) {
        const auto data = c->cast_vector<int64_t>();
        OPENVINO_ASSERT(element < data.size() && data[element] == expected,
                        "[TP] '", consumer->get_friendly_name(), "' expects ", expected, " at element ", element,
                        " of its shape input");
        annotate_shape_input(consumer, port, {element}, segments, block);
        return;
    }
    if (ov::is_type<ov::op::v0::Concat>(source)) {
        size_t offset = 0;
        for (size_t i = 0; i < source->get_input_size(); ++i) {
            const auto& shape = source->get_input_partial_shape(i);
            if (!shape.is_static() || shape.size() != 1)
                break;
            const auto length = static_cast<size_t>(shape[0].get_length());
            if (element < offset + length) {
                annotate_size(source, i, element - offset, expected, segments, block);
                return;
            }
            offset += length;
        }
    }
    OPENVINO_THROW("[TP] The GQA expansion at '", consumer->get_friendly_name(), "' takes its group size from ",
                   source->get_type_name(), " '", source->get_friendly_name(),
                   "', which cannot be given each rank's share of the group");
}

/// The axis where `node` expands a size-1 axis of input 0 to `group`.
int64_t expanded_axis(const std::shared_ptr<ov::Node>& node, size_t group) {
    const auto& in = node->get_input_partial_shape(0);
    const auto& out = node->get_output_partial_shape(0);
    if (in.rank().is_static() && out.rank().is_static() && in.size() == out.size()) {
        for (size_t a = 0; a < in.size(); ++a) {
            if (in[a].is_static() && in[a].get_length() == 1 && out[a].is_static() &&
                out[a].get_length() == static_cast<int64_t>(group))
                return static_cast<int64_t>(a);
        }
    }
    OPENVINO_THROW("[TP] Could not find the axis the GQA expansion '", node->get_friendly_name(), "' expands to ",
                   group, " query heads per KV head");
}

using MultiQueryBcst = decltype(ov::op::util::match_multi_query_bcst(nullptr));

/// With fewer KV heads than ranks, every rank expands its KV head to its share of the group only.
void annotate_group_expansion(const ov::pass::pattern::PatternValueMap& map,
                              const MultiQueryBcst& pattern,
                              size_t group,
                              const std::vector<Segment>& share,
                              const TPAnalyzerConfig& config,
                              const std::string& block) {
    const auto& computed_bcst = std::get<3>(pattern);
    const auto& multiply = std::get<4>(pattern);
    const auto& computed_bcst3 = std::get<5>(pattern);

    if (map.count(computed_bcst3)) {
        const auto bcst = map.at(computed_bcst3).get_node_shared_ptr();
        annotate_size(bcst, 1, static_cast<size_t>(expanded_axis(bcst, group)), static_cast<int64_t>(group), share, block);
        return;
    }
    OPENVINO_ASSERT(map.count(multiply), "[TP] Unexpected form of the GQA expansion");
    const auto mul = map.at(multiply).get_node_shared_ptr();
    if (map.count(computed_bcst)) {
        const auto bcst = map.at(computed_bcst).get_node_shared_ptr();
        const auto& out = bcst->get_output_partial_shape(0);
        for (size_t a = 0; a < out.size(); ++a) {
            if (out[a].is_static() && out[a].get_length() == static_cast<int64_t>(group)) {
                annotate_size(bcst, 1, a, static_cast<int64_t>(group), share, block);
                return;
            }
        }
        OPENVINO_THROW("[TP] Could not find the group axis of the GQA expansion '", bcst->get_friendly_name(), "'");
    }
    // Multiply by a tensor of ones shaped [.., group, ..]: every rank keeps its share of it.
    const auto& ones = mul->get_input_partial_shape(1);
    const auto axis = expanded_axis(mul, group) - static_cast<int64_t>(mul->get_output_partial_shape(0).size() - ones.size());
    OPENVINO_ASSERT(axis >= 0 && annotate_chain(mul->input(1), axis, share, ShardStrategy::COLUMN_PARALLEL, config.world_size, block),
                    "[TP] Could not give every rank its share of the GQA expansion '", mul->get_friendly_name(), "'");
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
                         ShardAnnotation heads(ShardStrategy::COLUMN_PARALLEL, -1, g.kv_heads_split,
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
            const auto& q_heads = g.q_heads;
            const auto& kv_heads = g.kv_heads_split;
            const auto group = static_cast<size_t>(g.heads / g.kv_heads);
            // A rank expands its KV head to its share of the group only.
            const std::vector<Segment> group_share{{group, group / g.replicas, 1, g.replicas}};

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
                        if (g.replicas > 1)
                            annotate_group_expansion(broadcast.get_pattern_value_map(), kv_bcst, group, group_share, config, block);
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

                    // Its initializer: Broadcast(0, Concat(.., Constant{kv_heads}, ..)), the KV-head axis at element 1.
                    if (read->get_input_size() == 0)
                        continue;
                    const auto init = read->get_input_node_shared_ptr(0);
                    if (!ov::is_type_any_of<ov::op::v1::Broadcast, ov::op::v3::Broadcast>(init))
                        continue;
                    annotate_size(init, 1, 1, g.kv_heads, kv_heads, block);
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
