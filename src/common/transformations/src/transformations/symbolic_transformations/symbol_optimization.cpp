// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/symbolic_transformations/symbol_optimization.hpp"

#include "itt.hpp"
#include "openvino/core/bound_evaluation_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/util/multi_subgraph_base.hpp"
#include "transformations/utils/utils.hpp"

namespace {
void update_symbol(std::shared_ptr<ov::Symbol>& symbol) {
    if (symbol != nullptr)
        symbol = ov::symbol::ancestor_of(symbol);
}

void apply_table_of_equivalence_on_model(const std::shared_ptr<ov::Model>& m) {
    for (const auto& op : m->get_ordered_ops()) {
        // handle inner sub-graphs
        if (auto multi_subgraph_op = std::dynamic_pointer_cast<ov::op::util::MultiSubGraphOp>(op))
            for (const auto& sub_graph : multi_subgraph_op->get_functions())
                if (sub_graph)
                    apply_table_of_equivalence_on_model(sub_graph);

        for (auto& output : op->outputs()) {
            // re-setting root symbols for shapes
            auto shape = output.get_partial_shape();
            for (auto& d : shape) {
                if (d.is_static())
                    continue;
                auto symbol = d.get_symbol();
                update_symbol(symbol);
                d.set_symbol(symbol);
            }
            op->set_output_type(output.get_index(), output.get_element_type(), shape);
            // re-setting root symbols for values
            auto value_symbols = output.get_tensor().get_value_symbol();
            for (auto& symbol : value_symbols)
                update_symbol(symbol);
            output.get_tensor().set_value_symbol(value_symbols);
        }
    }
}
}  // namespace

bool ov::pass::ApplySymbolEquivalence::run_on_model(const std::shared_ptr<ov::Model>& m) {
    RUN_ON_FUNCTION_SCOPE(ApplySymbolEquivalence);
    apply_table_of_equivalence_on_model(m);
    return false;
}

// Symbol to source map
using STS_map = std::unordered_map<std::shared_ptr<ov::Symbol>, ov::Output<ov::Node>>;

namespace {
int64_t get_idx_of_symbol_in_source(const ov::Output<ov::Node>& source, const std::shared_ptr<ov::Symbol>& symbol) {
    int64_t idx = -1;
    if (symbol == nullptr)
        return idx;
    auto pshape = source.get_partial_shape();
    auto rank = pshape.rank();
    if (rank.is_dynamic())
        return idx;
    for (int64_t i = 0; i < rank.get_length(); ++i) {
        auto s = pshape[i].get_symbol();
        if (ov::symbol::are_equal(s, symbol)) {
            idx = i;
            break;
        }
    }
    return idx;
}

ov::Output<ov::Node> alternative_source_from_existing_value(const std::shared_ptr<ov::Symbol>& symbol,
                                                            const ov::Output<ov::Node>& original_output,
                                                            STS_map& symbol_value_source) {
    auto alternative_source = ov::Output<ov::Node>();
    if (symbol_value_source.count(symbol)) {
        alternative_source = symbol_value_source[symbol];
        const auto &original_shape = original_output.get_shape(), &alternative_shape = alternative_source.get_shape();
        const auto &original_et = original_output.get_element_type(),
                   &alternative_et = alternative_source.get_element_type();
        if (alternative_shape != original_shape && (original_shape.empty() || original_shape == ov::Shape{0})) {
            auto squeeze = std::make_shared<ov::op::v0::Squeeze>(alternative_source);
            ov::copy_runtime_info(original_output.get_node_shared_ptr(), squeeze);
            alternative_source = squeeze->output(0);
        } else if (alternative_shape != original_shape) {
            auto shape = ov::op::v0::Constant::create(ov::element::i64, {original_shape.size()}, original_shape);
            auto reshape = std::make_shared<ov::op::v1::Reshape>(alternative_source, shape, false);
            ov::copy_runtime_info(original_output.get_node_shared_ptr(), reshape);
            alternative_source = reshape->output(0);
        }
        if (alternative_et != original_et) {
            auto convert = std::make_shared<ov::op::v0::Convert>(alternative_source, original_et);
            ov::copy_runtime_info(original_output.get_node_shared_ptr(), convert);
            alternative_source = convert->output(0);
        }
    }
    return alternative_source;
}

ov::Output<ov::Node> alternative_source_from_shape_source(const STS_map& symbol_shape_source,
                                                          const std::shared_ptr<ov::Symbol>& symbol,
                                                          const ov::Output<ov::Node>& original_output,
                                                          STS_map& symbol_value_source) {
    auto alternative_source = ov::Output<ov::Node>();
    if (symbol_shape_source.count(symbol)) {
        // replacing via constructing the symbol source and saving it for the future
        const auto& source = symbol_shape_source.at(symbol);
        const int64_t& idx = get_idx_of_symbol_in_source(source, symbol);
        if (idx == -1)
            return alternative_source;
        const auto& original_et = original_output.get_element_type();
        std::shared_ptr<ov::Node> shape;
        if (original_et == ov::element::i32 || original_et == ov::element::i64) {
            shape = std::make_shared<ov::op::v3::ShapeOf>(source, original_et);
        } else {
            shape = std::make_shared<ov::op::v3::ShapeOf>(source);
            ov::copy_runtime_info(original_output.get_node_shared_ptr(), shape);
            shape = std::make_shared<ov::op::v0::Convert>(shape, original_et);
        }
        auto indices = ov::op::v0::Constant::create(ov::element::i64, original_output.get_shape(), {idx});
        auto axis = ov::op::v0::Constant::create(ov::element::i64, {}, {0});
        auto gather = std::make_shared<ov::op::v8::Gather>(shape, indices, axis);
        ov::copy_runtime_info(original_output.get_node_shared_ptr(), {shape, indices, axis, gather});
        alternative_source = gather;
        symbol_value_source[symbol] = alternative_source;
    }
    return alternative_source;
}

ov::Output<ov::Node> get_alternative_source_from_value_or_shape_source(const STS_map& symbol_shape_source,
                                                                       const std::shared_ptr<ov::Symbol>& symbol,
                                                                       const ov::Output<ov::Node>& original_output,
                                                                       STS_map& symbol_value_source) {
    auto alternative_source = ov::Output<ov::Node>();
    if (symbol == nullptr)
        return alternative_source;
    alternative_source = alternative_source_from_existing_value(symbol, original_output, symbol_value_source);
    if (!alternative_source.get_node_shared_ptr())
        alternative_source =
            alternative_source_from_shape_source(symbol_shape_source, symbol, original_output, symbol_value_source);
    return alternative_source;
}

ov::Output<ov::Node> alternative_source_from_concat_input_sources(const STS_map& symbol_shape_source,
                                                                  const std::shared_ptr<ov::Symbol>& symbol,
                                                                  const ov::Output<ov::Node>& original_output,
                                                                  STS_map& symbol_value_source) {
    auto alternative_source = ov::Output<ov::Node>();
    if (symbol_shape_source.count(symbol)) {
        const auto& source = symbol_shape_source.at(symbol);
        auto concat = ov::as_type_ptr<ov::op::v0::Concat>(source.get_node_shared_ptr());
        if (!concat || concat->get_input_size() != 2)
            return alternative_source;
        int64_t idx = get_idx_of_symbol_in_source(source, symbol);
        if (idx == -1 || idx != concat->get_concatenation_axis())
            return alternative_source;
        // optimize using the knowledge of the Concat SI and what happens on the axis
        const auto& lhs_pshape = concat->get_input_partial_shape(0);
        const auto& rhs_pshape = concat->get_input_partial_shape(1);
        if (lhs_pshape.rank().is_static() && rhs_pshape.rank().is_static()) {
            auto lhs_symbol = lhs_pshape[idx].get_symbol();
            auto lhs_alternative = get_alternative_source_from_value_or_shape_source(symbol_shape_source,
                                                                                     lhs_symbol,
                                                                                     original_output,
                                                                                     symbol_value_source);

            auto rhs_symbol = rhs_pshape[idx].get_symbol();
            auto rhs_alternative = get_alternative_source_from_value_or_shape_source(symbol_shape_source,
                                                                                     rhs_symbol,
                                                                                     original_output,
                                                                                     symbol_value_source);

            if (lhs_alternative.get_node_shared_ptr() && rhs_alternative.get_node_shared_ptr()) {
                alternative_source = std::make_shared<ov::op::v1::Add>(lhs_alternative, rhs_alternative);
                ov::copy_runtime_info(original_output.get_node_shared_ptr(), alternative_source.get_node_shared_ptr());
                alternative_source.get_tensor().set_value_symbol({symbol});
                symbol_value_source[symbol] = alternative_source;
            }
        }
    }
    return alternative_source;
}

void optimize_value_usage(ov::Output<ov::Node>& output, STS_map& symbol_shape_source, STS_map& symbol_value_source) {
    auto value_symbols = output.get_tensor().get_value_symbol();
    if (value_symbols.size() != 1)
        return;
    auto symbol = value_symbols[0];
    if (symbol == nullptr)
        return;
    auto pshape = output.get_partial_shape();
    if (pshape.is_dynamic() || ov::shape_size(pshape.to_shape()) != 1)
        return;

    ov::Output<ov::Node> alternative_source =
        alternative_source_from_concat_input_sources(symbol_shape_source, symbol, output, symbol_value_source);
    if (!alternative_source.get_node_shared_ptr())
        alternative_source =
            get_alternative_source_from_value_or_shape_source(symbol_shape_source, symbol, output, symbol_value_source);

    if (alternative_source.get_node_shared_ptr() != nullptr) {
        evaluate_both_bounds(alternative_source);
        output.replace(alternative_source);
    } else {
        // in case we can not optimize it -- it is symbol which appeared just now on the value path
        symbol_value_source[symbol] = output;
    }
}

void save_shape_sources(const ov::Output<ov::Node>& output, STS_map& symbol_shape_source) {
    for (const auto& d : output.get_partial_shape()) {
        if (d.is_static())
            continue;
        auto symbol = d.get_symbol();
        if (symbol == nullptr)
            continue;
        if (symbol_shape_source.count(symbol))
            continue;
        symbol_shape_source[symbol] = output;
    }
}
}  // namespace

bool ov::pass::OptimizeSymbolsUsedAsValues::run_on_model(const std::shared_ptr<ov::Model>& m) {
    RUN_ON_FUNCTION_SCOPE(OptimizeSymbolsUsedAsValues);
    STS_map symbol_shape_source;
    STS_map symbol_value_source;
    for (const auto& op : m->get_ordered_ops()) {
        // Result has output port which has shared (during validate_and_infer_type) tensor with input port.
        // Transformations may replace input of Result. After replacement and before Result::validate_and_infer_type --
        // output tensor of Result may contain inaccurate shape / symbols due to the sharing with tensor which may be
        // already detached from the model. To avoid creating ShapeOf from Result in these cases we exclude it from this
        // optimization entirely
        if (auto result = ov::as_type_ptr<op::v0::Result>(op))
            continue;

        // LTS maps aren't shared with sub-graphs because inner graph can not access outer graph for label sources
        ov::op::util::process_subgraph(*this, op);

        for (auto& output : op->outputs()) {
            optimize_value_usage(output, symbol_shape_source, symbol_value_source);
            save_shape_sources(output, symbol_shape_source);
        }
    }
    return true;
}
