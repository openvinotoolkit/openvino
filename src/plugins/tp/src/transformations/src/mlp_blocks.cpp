// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <numeric>

#include "block_utils.hpp"
#include "openvino/core/except.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/op/util/unary_elementwise_arithmetic.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "tp_transformations/analyzer.hpp"

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

std::shared_ptr<ov::Node> activation(const std::shared_ptr<ov::Node>& input) {
    // wrap_type matches by is_castable, so the base class covers every activation derived from it.
    return wrap_type<ov::op::util::UnaryElementwiseArithmetic, ov::op::v4::Swish>({input});
}

/// Annotates an MLP: `columns` produce the intermediate dimension, `down` consumes it.
/// The split step is the coarsest any of the three weights needs (quantization groups,
/// packed sub-byte elements), so all of them are cut at the same offsets.
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

    const std::vector<Segment> segments{{static_cast<size_t>(size), static_cast<size_t>(size % unit == 0 ? unit : 1)}};
    for (const auto& column : columns) {
        if (!annotate_projection(column, ShardStrategy::COLUMN_PARALLEL, segments, Collective::NONE, config.world_size, block))
            report.runtime_slices.push_back(column->get_friendly_name());
    }
    if (!annotate_projection(down, ShardStrategy::ROW_PARALLEL, segments, Collective::ALL_REDUCE, config.world_size, block))
        report.runtime_slices.push_back(down->get_friendly_name());

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
        if (!annotate_projection(lm_head,
                                 ShardStrategy::COLUMN_PARALLEL,
                                 {{static_cast<size_t>(vocab), 1}},
                                 Collective::GATHER,
                                 config.world_size,
                                 "TPVocabParallelBlock"))
            report->runtime_slices.push_back(lm_head->get_friendly_name());
        report->lm_head = true;
        return false;
    });
}

}  // namespace tp
}  // namespace ov
