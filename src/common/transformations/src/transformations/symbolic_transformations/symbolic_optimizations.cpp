// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/symbolic_transformations/symbolic_optimizations.hpp"

#include "itt.hpp"
#include "openvino/core/descriptor_tensor.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/util/symbolic_info.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/visualize_tree.hpp"
#include "transformations/common_optimizations/dimension_tracking.hpp"
#include "transformations/common_optimizations/nop_elimination.hpp"
#include "transformations/common_optimizations/shared_ops_optimization.hpp"
#include "transformations/common_optimizations/simplify_shape_of_sub_graph.hpp"
#include "transformations/symbolic_transformations/chained_maximum.hpp"
#include "transformations/symbolic_transformations/dereshape_matmul.hpp"
#include "transformations/symbolic_transformations/nop_broadcast.hpp"
#include "transformations/symbolic_transformations/reshape_optimizations.hpp"
#include "transformations/symbolic_transformations/symbol_optimization.hpp"
#include "transformations/symbolic_transformations/utils.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov::pass;
using namespace ov::symbol::util;

namespace {
void symbolic_set_up_for_shape(ov::PartialShape& shape) {
    if (shape.rank().is_dynamic())
        return;
    for (auto& d : shape) {
        bool is_static = d.is_static(), has_symbol = d.has_symbol();
        if (is_static && has_symbol)
            d.set_symbol(nullptr);  // remove symbols from static dims on shapes to reduce symbol clutter
        if (is_static || has_symbol)
            continue;
        d.set_symbol(std::make_shared<ov::Symbol>());
    }
}

void special_case_range_symbol_propagation(const std::shared_ptr<ov::Node>& node) {
    /* Symbol propagation through specific Range operation
          start    shift
            |  \   /
            |   Add   step == 1
            \    /    /
               Range
    */
    if (!ov::is_type<ov::op::v0::Range>(node) && !ov::is_type<ov::op::v4::Range>(node))
        return;

    auto output_shape = node->get_output_partial_shape(0);
    if (output_shape.rank().is_dynamic() || output_shape.size() != 1)
        return;

    auto step_value = ov::util::get_constant_from_source(node->input_value(2));
    if (!step_value || step_value->cast_vector<int64_t>()[0] != 1)
        return;

    auto start_symbols = node->get_input_tensor(0).get_value_symbol();
    if (start_symbols.size() != 1 || start_symbols[0] == nullptr)
        return;
    auto start_symbol = start_symbols[0];

    auto stop_node = node->input_value(1).get_node_shared_ptr();
    if (!ov::is_type<ov::op::v1::Add>(stop_node))
        return;
    auto add_in0_symbols = stop_node->get_input_tensor(0).get_value_symbol();
    if (add_in0_symbols.size() != 1 || add_in0_symbols[0] == nullptr)
        return;
    auto add_in0_symbol = add_in0_symbols[0];

    auto add_in1_symbols = stop_node->get_input_tensor(1).get_value_symbol();
    if (add_in1_symbols.size() != 1 || add_in1_symbols[0] == nullptr)
        return;
    auto add_in1_symbol = add_in1_symbols[0];

    if (add_in0_symbol == start_symbol)
        output_shape[0].set_symbol(add_in1_symbol);
    else if (add_in1_symbol == start_symbol)
        output_shape[0].set_symbol(add_in0_symbol);
    node->set_output_type(0, node->get_output_element_type(0), output_shape);
}
}  // namespace

bool ov::pass::SymbolicPropagation::run_on_model(const std::shared_ptr<ov::Model>& m) {
    RUN_ON_MODEL_SCOPE(SymbolicPropagation);

    for (const auto& op : m->get_ordered_ops()) {
        // since we disable invalidation with the following two lines, we have to invalidate manually here
        op->invalidate_values();
        for (auto& output : op->outputs())
            ov::skip_invalidation(output);
        op->revalidate_and_infer_types();
        // Recursively apply transformation for sub-graph based operations
        ov::op::util::process_subgraph(*this, op);

        // additional symbol propagation rules must be triggered here
        special_case_range_symbol_propagation(op);
        // additional symbol propagation rules must be triggered here

        for (auto& output : op->outputs()) {
            auto shape = output.get_partial_shape();
            symbolic_set_up_for_shape(shape);
            ov::descriptor::set_tensor_type(output.get_tensor(), output.get_element_type(), shape);
        }
    }
    return true;
}

ov::pass::LabelResolvingThroughSelect::LabelResolvingThroughSelect() {
    MATCHER_SCOPE(LabelResolvingThroughSelect);
    auto add = pattern::wrap_type<op::util::BinaryElementwiseArithmetic>();
    auto input_reshape = pattern::wrap_type<op::v1::Reshape>({add, pattern::any_input()});

    auto select_then = pattern::wrap_type<op::v1::Select>({pattern::any_input(), input_reshape, pattern::any_input()});
    auto select_else = pattern::wrap_type<op::v1::Select>({pattern::any_input(), pattern::any_input(), input_reshape});
    auto select = std::make_shared<pass::pattern::op::Or>(OutputVector{select_then, select_else});

    auto softmax = pattern::wrap_type<op::v1::Softmax>({select});
    auto reshape = pattern::wrap_type<op::v1::Reshape>({softmax, pattern::any_input()});

    ov::matcher_pass_callback matcher_pass_callback = [=](pattern::Matcher& m) {
        const auto& value_map = m.get_pattern_value_map();
        ov::TensorSymbol reshape_symbols, add_0_symbols, add_1_symbols;
        if (!get_symbols(value_map.at(reshape).get_partial_shape(), reshape_symbols))
            return false;
        auto add_node = value_map.at(add).get_node_shared_ptr();
        auto add_0_pshape = add_node->input_value(0).get_partial_shape();
        auto add_1_pshape = add_node->input_value(1).get_partial_shape();
        if (!get_symbols(add_0_pshape, add_0_symbols) && !get_symbols(add_1_pshape, add_1_symbols))
            return false;

        if (are_unique_and_equal_symbols(reshape_symbols, add_0_symbols)) {
            // we detected that no broadcasting was done during binary elementwise and select, propagating symbols
            // through
            add_node->set_output_type(0, add_node->get_output_element_type(0), add_0_pshape);
        } else if (are_unique_and_equal_symbols(reshape_symbols, add_1_symbols)) {
            // we detected that no broadcasting was done during binary elementwise and select, propagating symbols
            // through
            add_node->set_output_type(0, add_node->get_output_element_type(0), add_1_pshape);
        } else {
            return false;
        }

        std::shared_ptr<ov::Node> select_node = nullptr;
        if (value_map.count(select_then))
            select_node = value_map.at(select_then).get_node_shared_ptr();
        if (value_map.count(select_else))
            select_node = value_map.at(select_else).get_node_shared_ptr();
        if (select_node == nullptr)
            return false;

        auto select_output = select_node->output(0);
        const auto& reshape_pshape = value_map.at(input_reshape).get_partial_shape();
        select_node->set_output_type(0, select_node->get_output_element_type(0), reshape_pshape);
        value_map.at(softmax).get_node_shared_ptr()->validate_and_infer_types();
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(reshape, matcher_name);
    register_matcher(m, matcher_pass_callback);
}

ov::pass::SymbolicOptimizations::SymbolicOptimizations(bool full_run) {
    m_manager = std::make_shared<pass::Manager>("Symbolic");
    m_manager->set_per_pass_validation(false);

#define REGISTER_SYMBOLIC(region, ...) m_manager->register_pass<region>(__VA_ARGS__);

    REGISTER_SYMBOLIC(SymbolicPropagation)
    if (full_run) {
        // symbolic based transformations allowing for better static dimension propagation
        REGISTER_SYMBOLIC(ChainedMaximumOptimization)
        REGISTER_SYMBOLIC(NopBroadcast)
        // regular transformations which are needed right now since they clean up unnecessary operations
        REGISTER_SYMBOLIC(NopElimination)        // Broadcast (Tile) Ones + Remove Slice Before GatherElements
        REGISTER_SYMBOLIC(SharedOpOptimization)  // Shared GatherElements
    }
    // transformations which use symbols for optimizations
    REGISTER_SYMBOLIC(ApplySymbolEquivalence)
    if (full_run) {
        REGISTER_SYMBOLIC(OptimizeSymbolsUsedAsValues)  // reduce shape sub-graphs
        REGISTER_SYMBOLIC(LabelResolvingThroughSelect)  // figures out that broadcasting didn't happen through Select op
        REGISTER_SYMBOLIC(DeReshapeMatMul)
        REGISTER_SYMBOLIC(DeReshapeFullyConnected)
        REGISTER_SYMBOLIC(ReshapeOptimizations)
        REGISTER_SYMBOLIC(SimplifyShapeOfSubGraph)
    }
}

bool ov::pass::SymbolicOptimizations::run_on_model(const std::shared_ptr<ov::Model>& m) {
    RUN_ON_FUNCTION_SCOPE(SymbolicOptimizations);

    // Eliminate Squeeze/Unsqueeze might convert Squeeze/Unsqueeze ops to Reshape
    // it may break NNCF patterns and lead to unexpected FakeQuantize ops in the model.
    // So we decided to disable these passes in SymbolicOptimizations.
    const auto& pass_config = m_manager->get_pass_config();
    pass_config->disable<EliminateSqueeze>();
    pass_config->disable<EliminateUnsqueeze>();

    m_manager->run_passes(m);
    ov::remove_skip_invalidation_rti(m);
    return true;
}
