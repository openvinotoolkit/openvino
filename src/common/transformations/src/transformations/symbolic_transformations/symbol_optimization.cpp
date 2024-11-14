// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/symbolic_transformations/symbol_optimization.hpp"

#include "itt.hpp"
#include "openvino/core/bound_evaluation_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/tensor_util.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/util/multi_subgraph_base.hpp"
#include "openvino/op/util/op_types.hpp"
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
        if (idx == -1)
            return alternative_source;
        const auto rank = source.get_partial_shape().rank().get_length();
        if (idx != ov::util::normalize(concat->get_axis(), rank))
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
        ov::util::evaluate_both_bounds(alternative_source);
        output.replace(alternative_source);
    } else {
        // in case we can not optimize it -- it is symbol which appeared just now on the value path
        symbol_value_source[symbol] = output;
    }
}

std::vector<std::shared_ptr<ov::Node>> topological_order(const std::shared_ptr<ov::Model>& m) {
    auto order = m->get_ordered_ops();

    // step 1: split model into parameter related and parameter non-related ops
    const std::string op_depends_on_parameter = "topological_sort_op_depends_on";
    // values: true - parameter dependent; false otherwise
    for (const auto& op : order) {
        if (ov::as_type_ptr<ov::op::v0::Parameter>(op)) {
            op->get_rt_info()[op_depends_on_parameter] = true;
        } else if (ov::as_type_ptr<ov::op::v0::Constant>(op) || ov::as_type_ptr<ov::op::v0::ShapeOf>(op) ||
                   ov::as_type_ptr<ov::op::v3::ShapeOf>(op) ||
                   std::dynamic_pointer_cast<ov::op::util::VariableExtension>(op)) {
            op->get_rt_info()[op_depends_on_parameter] = false;
        } else {  // deduce op type from inputs
            const auto& inputs = op->input_values();
            op->get_rt_info()[op_depends_on_parameter] =
                std::any_of(inputs.begin(),
                            inputs.end(),
                            [&op_depends_on_parameter](const ov::Output<ov::Node>& input) {
                                return input.get_node_shared_ptr()->get_rt_info()[op_depends_on_parameter].as<bool>();
                            });
        }
    }
    // step 2: starting from Result -- assign weight to ops:
    //      if parameter dependant, weights is maximum of output indices plus one
    //      else weights is maximum of output indices
    // this step doesn't assign weights to all the ops, this is intentional and will be used in the following step
    const std::string weight_rt_info_name = "topological_sort_weight";
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        const auto& op = *it;
        int64_t weight = 0;
        if (ov::as_type_ptr<ov::op::v0::Result>(op)) {
            op->get_rt_info()[weight_rt_info_name] = weight;
        } else {
            bool output_has_weight = false;
            for (const auto& output : op->outputs()) {
                for (const auto& input : output.get_target_inputs()) {
                    const auto& output_op = input.get_node();
                    const auto& rt_info = output_op->get_rt_info();
                    if (!rt_info.count(weight_rt_info_name))
                        continue;
                    output_has_weight = true;
                    auto output_weight = rt_info.at(weight_rt_info_name).as<int64_t>();
                    weight = output_weight > weight ? output_weight : weight;
                }
            }
            if (output_has_weight) {
                if (op->get_rt_info()[op_depends_on_parameter].as<bool>()) {
                    weight += 1;
                }
                op->get_rt_info()[weight_rt_info_name] = weight;
            }
        }
    }
    // step 3: make propagation for all the nodes:
    // if weight is already assigned -- skip operation
    // else operation weights is minimum of input indices
    // if all operation inputs have no weights -- this op is isolated and this algorithm doesn't make sense,
    // such cases are extremely rare and rather theoretical, to handle them we return original ov::Model op order
    std::map<int64_t, std::vector<std::shared_ptr<ov::Node>>> level_to_vector;
    for (const auto& op : order) {
        if (!op->get_rt_info().count(weight_rt_info_name)) {
            int64_t weight = std::numeric_limits<int64_t>::max();
            for (const auto& input : op->input_values()) {
                const auto& rt_info = input.get_node_shared_ptr()->get_rt_info();
                if (!rt_info.count(weight_rt_info_name))
                    continue;
                auto input_weight = rt_info.at(weight_rt_info_name).as<int64_t>();
                weight = input_weight < weight ? input_weight : weight;
            }
            if (weight != std::numeric_limits<int64_t>::max())
                op->get_rt_info()[weight_rt_info_name] = weight;
            else
                return m->get_ordered_ops();
        }
        level_to_vector[op->get_rt_info().at(weight_rt_info_name).as<int64_t>()].push_back(op);
    }
    // finalization: descending order for levels and ops within level are ordered by get_ordered_ops
    std::vector<std::shared_ptr<ov::Node>> result;
    result.reserve(order.size());
    for (auto it = level_to_vector.rbegin(); it != level_to_vector.rend(); ++it) {
        const auto& item = *it;
        result.insert(result.end(), item.second.begin(), item.second.end());
        for (const auto& op : item.second) {
            op->get_rt_info().erase(weight_rt_info_name);
            op->get_rt_info().erase(op_depends_on_parameter);
        }
    }
    return result;
}

void save_shape_sources(const std::shared_ptr<ov::Node>& op, STS_map& symbol_shape_source) {
    if (ov::is_type<ov::op::v0::ShapeOf>(op) || ov::is_type<ov::op::v3::ShapeOf>(op)) {
        const auto& output = op->input_value(0);
        if (output.get_partial_shape().rank().is_dynamic())
            return;
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
    } else if (const auto concat = ov::as_type_ptr<ov::op::v0::Concat>(op)) {
        // we agreed to detach ShapeOf sourcing output axis dimension from Concat operations for smoother matching
        // this code allows to find places to reconnect ShapeOfs
        const auto rank = concat->get_output_partial_shape(0).rank();
        auto axis = concat->get_axis();
        if (axis < 0) {
            if (rank.is_dynamic())
                return;
            axis += rank.get_length();
        }
        for (const auto& input : concat->input_values()) {
            if (input.get_partial_shape().rank().is_dynamic())
                continue;
            const auto dimension = input.get_partial_shape()[axis];
            if (dimension.is_static())
                continue;
            auto symbol = dimension.get_symbol();
            if (symbol == nullptr)
                continue;
            if (symbol_shape_source.count(symbol))
                continue;
            symbol_shape_source[symbol] = input;
        }
    }
}

struct OutputValue {
    std::vector<ov::Any> value;

    bool operator==(const OutputValue& other) const {
        return value == other.value;
    }

    bool operator<(const OutputValue& other) const {
        return std::lexicographical_compare(
            std::begin(value),
            std::end(value),
            std::begin(other.value),
            std::end(other.value),
            [](const ov::Any& a, const ov::Any& b) {
                // each element is either a symbol or an integer. in case they differ any integer is less than a symbol.
                if (a.is<std::shared_ptr<ov::Symbol>>() && b.is<std::shared_ptr<ov::Symbol>>())
                    return a.as<std::shared_ptr<ov::Symbol>>() < b.as<std::shared_ptr<ov::Symbol>>();
                if (a.is<int64_t>() && b.is<int64_t>())
                    return a.as<int64_t>() < b.as<int64_t>();
                return a.is<int64_t>();
            });
    }

    static ov::optional<OutputValue> make(const ov::Output<ov::Node>& output) {
        auto symbols = output.get_tensor().get_value_symbol();
        if (symbols.empty() || symbols.size() == 1)
            return {};

        const auto& lower_value = ov::util::to_vector<int64_t>(output.get_tensor().get_lower_value());
        const auto& upper_value = ov::util::to_vector<int64_t>(output.get_tensor().get_upper_value());
        const auto& et = output.get_element_type();
        bool use_values = lower_value && upper_value && (et == ov::element::i64 || et == ov::element::i32);

        std::vector<ov::Any> symbols_as_any(symbols.size(), nullptr);
        for (size_t i = 0; i < symbols_as_any.size(); ++i) {
            if (use_values && lower_value->at(i) == upper_value->at(i))
                symbols_as_any[i] = lower_value->at(i);
            else if (symbols.at(i) != nullptr)
                symbols_as_any[i] = ov::symbol::ancestor_of(symbols.at(i));
            else
                return {};
        }
        return {OutputValue{std::move(symbols_as_any)}};
    }
};

void save_and_update_value_sources(const std::shared_ptr<ov::Node>& op,
                                   std::map<OutputValue, ov::Output<ov::Node>>& multi_symbol_source) {
    for (auto& output : op->outputs()) {
        if (output.get_tensor().get_value_symbol().size() < 2)
            continue;  // singular values are handled by optimize_value_usage helper

        if (auto result = OutputValue::make(output)) {
            if (multi_symbol_source.count(*result)) {
                auto alternative_source = multi_symbol_source[*result];
                if (output.get_element_type() != alternative_source.get_element_type()) {
                    auto convert = std::make_shared<ov::op::v0::Convert>(alternative_source, output.get_element_type());
                    ov::copy_runtime_info(output.get_node_shared_ptr(), convert);
                    alternative_source = convert->output(0);
                }
                if (output.get_partial_shape().is_dynamic() ||
                    output.get_partial_shape() != alternative_source.get_partial_shape())
                    continue;
                output.replace(alternative_source);
            } else {
                multi_symbol_source[*result] = output;
            }
        }
    }
}

}  // namespace

bool ov::pass::OptimizeSymbolsUsedAsValues::run_on_model(const std::shared_ptr<ov::Model>& m) {
    RUN_ON_FUNCTION_SCOPE(OptimizeSymbolsUsedAsValues);
    STS_map symbol_shape_source;
    STS_map symbol_value_source;
    std::map<OutputValue, ov::Output<ov::Node>> multi_symbol_source;
    for (const auto& op : topological_order(m)) {
        // Result has output port which has shared (during validate_and_infer_type) tensor with input port.
        // Transformations may replace input of Result. After replacement and before Result::validate_and_infer_type --
        // output tensor of Result may contain inaccurate shape / symbols due to the sharing with tensor which may be
        // already detached from the model. To avoid creating ShapeOf from Result in these cases we exclude it from this
        // optimization entirely
        if (auto result = ov::as_type_ptr<op::v0::Result>(op))
            continue;

        // LTS maps aren't shared with sub-graphs because inner graph can not access outer graph for label sources
        ov::op::util::process_subgraph(*this, op);

        for (auto& output : op->outputs())
            optimize_value_usage(output, symbol_shape_source, symbol_value_source);
        save_shape_sources(op, symbol_shape_source);
        save_and_update_value_sources(op, multi_symbol_source);
    }
    return true;
}
