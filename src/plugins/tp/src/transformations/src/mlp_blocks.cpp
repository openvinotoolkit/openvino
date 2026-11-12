// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <numeric>
#include <unordered_set>

#include "block_utils.hpp"
#include "openvino/core/except.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/op/util/unary_elementwise_arithmetic.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "tp_transformations/analyzer.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {
namespace tp {

using ov::pass::operator|;

namespace {

using namespace blocks;
using namespace ov::pass::pattern;

/// MatMul or MatMul + bias Add.
std::shared_ptr<ov::Node> with_bias(const std::shared_ptr<ov::Node>& matmul) {
    return matmul | wrap_type<ov::op::v1::Add>({matmul, any_input()});
}

/// Bytes of the constants behind `matmul`'s weight when another reader uses that weight too (a tied
/// embedding), 0 otherwise.
size_t shared_weight_bytes(const std::shared_ptr<ov::Node>& matmul) {
    std::unordered_set<ov::Node*> chain;
    ov::op::util::visit_path(
        matmul->get_input_node_ptr(1),
        chain,
        [](ov::Node*) {},
        [](ov::Node*) {
            return false;
        });
    bool shared = false;
    size_t bytes = 0;
    for (auto* node : chain) {
        const auto* constant = ov::as_type<ov::op::v0::Constant>(node);
        // Shape constants (Reshape targets) may be shared by unrelated ops.
        if (constant && (constant->get_element_type() == ov::element::i64 || constant->get_element_type() == ov::element::i32))
            continue;
        for (const auto& output : node->outputs()) {
            for (const auto& target : output.get_target_inputs())
                shared = shared || (target.get_node() != matmul.get() && chain.count(target.get_node()) == 0);
        }
        if (constant)
            bytes += constant->get_byte_size();
    }
    return shared ? bytes : 0;
}

std::shared_ptr<ov::Node> activation(const std::shared_ptr<ov::Node>& input) {
    // wrap_type matches by is_castable, so the base class covers every activation derived from it.
    return wrap_type<ov::op::util::UnaryElementwiseArithmetic, ov::op::v4::Swish>({input});
}

/// The intermediate dimension of `down`'s MLP in whole units of `unit`, the same number on every rank.
Segment intermediate_split(int64_t size,
                           int64_t unit,
                           const std::shared_ptr<ov::Node>& down,
                           size_t world_size,
                           const std::string& block) {
    if (size % unit != 0)
        unit = 1;
    const int64_t units = size / unit;
    OPENVINO_ASSERT(units % static_cast<int64_t>(world_size) == 0,
                    "[TP] '", down->get_friendly_name(), "' (", block, "): world size ", world_size,
                    " does not divide the MLP's intermediate size ", size, " evenly",
                    unit > 1 ? " in quantization groups of " + std::to_string(unit) : std::string{}, " (", units,
                    " unit(s)). Every rank takes an equal share of the MLP; use a number of devices that divides ",
                    units, ".");
    return {static_cast<size_t>(size), static_cast<size_t>(unit)};
}

/// Annotates an MLP: `columns` produce the intermediate dimension, `down` consumes it.
/// The split step is the coarsest any of the three weights needs (quantization groups),
/// so all of them are cut at the same offsets.
void annotate_mlp(const std::vector<std::shared_ptr<ov::Node>>& columns,
                  const std::shared_ptr<ov::Node>& down,
                  const TPAnalyzerConfig& config,
                  TPAnalysisReport& report,
                  const std::string& block) {
    const int64_t size = weight_dim(down, weight_axes(down).in_features);
    OPENVINO_ASSERT(size > 0, "[TP] '", down->get_friendly_name(), "' has a dynamic intermediate dimension");

    int64_t unit = 1;
    auto fold = [&](const std::shared_ptr<ov::Node>& matmul, int64_t axis) {
        const int64_t step = chain_granularity(matmul->input_value(1), axis, size);
        if (step > 0 && size % step == 0)
            unit = std::lcm(unit, step);
    };
    for (const auto& column : columns)
        fold(column, weight_axes(column).out_features);
    fold(down, weight_axes(down).in_features);

    const std::vector<Segment> segments{intermediate_split(size, unit, down, config.world_size, block)};
    for (const auto& column : columns)
        annotate_projection(column, ShardStrategy::COLUMN_PARALLEL, segments, Collective::NONE, config.world_size, block);
    annotate_projection(down, ShardStrategy::ROW_PARALLEL, segments, Collective::ALL_REDUCE, config.world_size, block);

    ++report.mlp_blocks;
    if (report.intermediate_size == 0)
        report.intermediate_size = size;
}

bool claimed(const std::shared_ptr<ov::Node>& node) {
    return get_role(*node) != nullptr;
}

}  // namespace

TPGatedMLPBlock::TPGatedMLPBlock(const TPAnalyzerConfig& config, TPAnalysisReportPtr report) {
    auto src = any_input();
    auto gate = wrap_type<ov::op::v0::MatMul>({src, any_input()});
    auto up = wrap_type<ov::op::v0::MatMul>({src, any_input()});
    auto act = activation(with_bias(gate));
    auto up_out = with_bias(up);
    auto mul = wrap_type<ov::op::v1::Multiply>({act, up_out}) | wrap_type<ov::op::v1::Multiply>({up_out, act});
    auto down = wrap_type<ov::op::v0::MatMul>({mul, any_input()});

    register_matcher(std::make_shared<Matcher>(down, get_type_info().name), [=](Matcher& m) {
        const auto& map = m.get_pattern_value_map();
        const auto d = map.at(down).get_node_shared_ptr();
        const auto g = map.at(gate).get_node_shared_ptr();
        const auto u = map.at(up).get_node_shared_ptr();
        if (!is_linear_layer(d.get()) || !is_linear_layer(g.get()) || !is_linear_layer(u.get()) ||
            g->input_value(0) != u->input_value(0) || claimed(d) || claimed(g) || claimed(u))
            return false;
        annotate_mlp({g, u}, d, config, *report, "TPGatedMLPBlock");
        return false;
    });
}

TPFusedGateUpMLPBlock::TPFusedGateUpMLPBlock(const TPAnalyzerConfig& config, TPAnalysisReportPtr report) {
    auto gate = any_input();
    auto up = any_input();
    auto act = activation(gate);
    auto mul = wrap_type<ov::op::v1::Multiply>({act, up}) | wrap_type<ov::op::v1::Multiply>({up, act});
    auto down = wrap_type<ov::op::v0::MatMul>({mul, any_input()});

    register_matcher(std::make_shared<Matcher>(down, get_type_info().name), [=](Matcher& m) {
        const auto& map = m.get_pattern_value_map();
        const auto d = map.at(down).get_node_shared_ptr();
        const auto g = map.at(gate);
        const auto u = map.at(up);
        // Both parts split off one projection's output.
        const auto split = g.get_node_shared_ptr();
        if (!is_linear_layer(d.get()) || claimed(d) || split->get_input_size() == 0)
            return false;
        const auto fused_output = split->input_value(0);
        const auto producers = nearest_producer_matmuls(fused_output);
        if (producers.size() != 1 || claimed(producers.front()) || projection_output(producers.front()) != fused_output)
            return false;
        const auto parts = feature_parts(fused_output);
        if (parts.size() != 2 || (parts[0] != g && parts[1] != g) || (parts[0] != u && parts[1] != u) || g == u)
            return false;

        const auto gate_up = producers.front();
        const int64_t size = weight_dim(d, weight_axes(d).in_features);
        for (const auto& part : parts) {
            const auto& shape = part.get_partial_shape();
            const auto& dim = shape[shape.rank().get_length() - 1];
            OPENVINO_ASSERT(dim.is_static() && dim.get_length() == size,
                            "[TP] '", gate_up->get_friendly_name(), "' is split into parts of ", dim, " features where '",
                            d->get_friendly_name(), "' takes ", size);
        }
        int64_t unit = 1;
        for (const auto& [matmul, axis, full] : {std::tuple{gate_up, weight_axes(gate_up).out_features, 2 * size},
                                                 std::tuple{d, weight_axes(d).in_features, size}}) {
            const int64_t step = chain_granularity(matmul->input_value(1), axis, full);
            if (step > 0 && size % step == 0)
                unit = std::lcm(unit, step);
        }
        const std::string block = "TPFusedGateUpMLPBlock";
        const Segment segment = intermediate_split(size, unit, d, config.world_size, block);
        annotate_projection(gate_up, ShardStrategy::COLUMN_PARALLEL, {segment, segment}, Collective::NONE, config.world_size, block);
        annotate_feature_parts(fused_output, {segment, segment}, block);
        annotate_projection(d, ShardStrategy::ROW_PARALLEL, {segment}, Collective::ALL_REDUCE, config.world_size, block);
        ++report->mlp_blocks;
        if (report->intermediate_size == 0)
            report->intermediate_size = size;
        return false;
    });
}

TPDenseMLPBlock::TPDenseMLPBlock(const TPAnalyzerConfig& config, TPAnalysisReportPtr report) {
    auto up = wrap_type<ov::op::v0::MatMul>({any_input(), any_input()});
    auto down = wrap_type<ov::op::v0::MatMul>({activation(with_bias(up)), any_input()});

    register_matcher(std::make_shared<Matcher>(down, get_type_info().name), [=](Matcher& m) {
        const auto& map = m.get_pattern_value_map();
        const auto d = map.at(down).get_node_shared_ptr();
        const auto u = map.at(up).get_node_shared_ptr();
        if (!is_linear_layer(d.get()) || !is_linear_layer(u.get()) || claimed(d) || claimed(u))
            return false;
        annotate_mlp({u}, d, config, *report, "TPDenseMLPBlock");
        return false;
    });
}

TPVocabParallelBlock::TPVocabParallelBlock(const TPAnalyzerConfig& config, TPAnalysisReportPtr report) {
    auto matmul = wrap_type<ov::op::v0::MatMul>({any_input(), any_input()});
    auto convert = wrap_type<ov::op::v0::Convert>({matmul});
    auto result = wrap_type<ov::op::v0::Result>({matmul | convert});

    register_matcher(std::make_shared<Matcher>(result, get_type_info().name), [=](Matcher& m) {
        const auto lm_head = m.get_pattern_value_map().at(matmul).get_node_shared_ptr();
        // Plain form only: a bias or a second reader leaves it whole.
        if (!is_linear_layer(lm_head.get()) || claimed(lm_head) || lm_head->output(0).get_target_inputs().size() != 1)
            return false;
        const int64_t vocab = weight_dim(lm_head, weight_axes(lm_head).out_features);
        // The gather assumes equal slices.
        if (vocab <= 0 || vocab % static_cast<int64_t>(config.world_size) != 0)
            return false;
        const std::vector<Segment> bands{{static_cast<size_t>(vocab), 1}};
        // A weight that cannot be cut at compile time leaves it whole as well.
        if (!chain_cut_problem(lm_head->input(1), weight_axes(lm_head).out_features, bands, config.world_size).empty())
            return false;
        // Read before the split gives the LM head a copy of the chain to cut.
        report->lm_head_copy_bytes = shared_weight_bytes(lm_head) / config.world_size;
        annotate_projection(lm_head, ShardStrategy::COLUMN_PARALLEL, bands, Collective::GATHER, config.world_size, "TPVocabParallelBlock");
        report->lm_head = true;
        return false;
    });
}

}  // namespace tp
}  // namespace ov
