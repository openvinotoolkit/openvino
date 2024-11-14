// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lir_comparator.hpp"

#include "common_test_utils/graph_comparator.hpp"
#include "snippets/lowered/pass/pass.hpp"
#include "common_test_utils/common_utils.hpp"

using namespace ov::snippets::lowered;

namespace std {
template <typename T>
inline string to_string(const vector<T>& vec) {
    return ov::test::utils::vec2str<T>(vec);
}

inline string to_string(const ov::snippets::Reg& reg) {
    return string("Reg(type = " + ov::snippets::regTypeToStr(reg.type) + ", idx = " + to_string(reg.idx) + ")");
}

inline string to_string(const ov::Node::type_info_t& info) {
    stringstream ss;
    ss << info;
    return ss.str();
}

inline string to_string(const SpecificLoopIterType& type) {
    stringstream ss;
    ss << type;
    return ss.str();
}
} // namespace std

namespace ov {
namespace test {
namespace snippets {
#define COMPARE(error_msg, lhs, rhs) \
    if (lhs != rhs)                  \
        return Result::error(std::string(error_msg) + " are different: " + std::to_string(lhs) + " and " + std::to_string(rhs))

#define PROPAGATE_ERROR(prefix, result) \
    if (!result.valid)                  \
        return Result::error(std::string(prefix) + ": " + result.message)

LIRComparator::Result LIRComparator::compare(const LinearIRPtr& linear_ir,
                                             const LinearIRPtr& linear_ir_ref) {
    OPENVINO_ASSERT(m_nodes_cmp_values != NodesCmpValues::NONE || m_lir_cmp_values != LIRCmpValues::NONE,
                    "Comparator mustn't be called with NONE cmp values");
    const auto& ops = linear_ir->get_ops();
    const auto& ops_ref = linear_ir_ref->get_ops();
    COMPARE("Number of ops", ops.size(), ops_ref.size());
    const auto& parameters = linear_ir->get_parameters();
    const auto& parameters_ref = linear_ir_ref->get_parameters();
    COMPARE("Number of parameters", parameters.size(), parameters_ref.size());
    const auto& results = linear_ir->get_results();
    const auto& results_ref = linear_ir_ref->get_results();
    COMPARE("Number of results", results.size(), results_ref.size());
    const auto& buffers = linear_ir->get_buffers();
    const auto& buffers_ref = linear_ir_ref->get_buffers();
    COMPARE("Number of buffers", buffers.size(), buffers_ref.size());

    auto run_comparison = [&](const ExpressionPtr& expr, const ExpressionPtr& expr_ref) {
        const auto node = expr->get_node();
        const auto node_ref = expr_ref->get_node();
        if (m_nodes_cmp_values != NodesCmpValues::NONE)
            PROPAGATE_ERROR("", Comparator(m_nodes_cmp_values).compare(node.get(), node_ref.get()));

        const std::string err_prefix = "Comparison failed for nodes " + node->get_friendly_name() + ", " + node_ref->get_friendly_name() + ". ";
        if (should_compare(LIRCmpValues::LOOP_INDICES))
            COMPARE(err_prefix + "Loop indices", expr->get_loop_ids(), expr_ref->get_loop_ids());

        if (should_compare(LIRCmpValues::PORT_DESCRIPTORS)) {
            PROPAGATE_ERROR(err_prefix + "Input descsriptors", compare_descs(expr->get_input_port_descriptors(), expr_ref->get_input_port_descriptors()));
            PROPAGATE_ERROR(err_prefix + "Output descsriptors", compare_descs(expr->get_output_port_descriptors(), expr_ref->get_output_port_descriptors()));
        }
        if (should_compare(LIRCmpValues::PORT_CONNECTORS)) {
            const auto& in_connectors = expr->get_input_port_connectors();
            const auto& in_connectors_ref = expr_ref->get_input_port_connectors();
            PROPAGATE_ERROR(err_prefix + "Input connectors", compare_port_connectors(in_connectors, in_connectors_ref));
            const auto& out_connectors = expr->get_output_port_connectors();
            const auto& out_connectors_ref = expr_ref->get_output_port_connectors();
            PROPAGATE_ERROR(err_prefix + "Output connectors", compare_port_connectors(out_connectors, out_connectors_ref));
        }
        return Result::ok();
    };

    for (auto param_it = parameters.begin(), param_it_ref = parameters_ref.begin(); param_it != parameters.end(); ++param_it, ++param_it_ref)
        PROPAGATE_ERROR("", run_comparison(*param_it, *param_it_ref));
    for (auto expr_it = ops.begin(), expr_it_ref = ops_ref.begin(); expr_it != ops.end(); ++expr_it, ++expr_it_ref)
        PROPAGATE_ERROR("", run_comparison(*expr_it, *expr_it_ref));
    for (auto result_it = results.begin(), result_it_ref = results_ref.begin(); result_it != results.end(); ++result_it, ++result_it_ref)
        PROPAGATE_ERROR("", run_comparison(*result_it, *result_it_ref));

    if (should_compare(LIRCmpValues::LOOP_MANAGER)) {
        PROPAGATE_ERROR("Loop managers", compare_loop_managers(linear_ir->get_loop_manager(), linear_ir_ref->get_loop_manager()));
    }
    return Result::ok();
}

LIRComparator::Result LIRComparator::compare_descs(const std::vector<PortDescriptorPtr>& descs, const std::vector<PortDescriptorPtr>& descs_ref) {
    COMPARE("Descriptors number", descs.size(), descs_ref.size());
    for (size_t i = 0; i < descs.size(); ++i) {
        COMPARE("Shapes", descs[i]->get_shape(), descs_ref[i]->get_shape());
        COMPARE("Layouts", descs[i]->get_layout(), descs_ref[i]->get_layout());
        COMPARE("Subtensors", descs[i]->get_subtensor(), descs_ref[i]->get_subtensor());
        COMPARE("Registers", descs[i]->get_reg(), descs_ref[i]->get_reg());
    }
    return Result::ok();
}

LIRComparator::Result LIRComparator::compare_loop_managers(const LoopManagerPtr& loop_manager,
                                                           const LoopManagerPtr& loop_manager_ref) {
    const auto& map = loop_manager->get_map();
    const auto& map_ref = loop_manager_ref->get_map();
    COMPARE("Loops map size", map.size(), map_ref.size());

    for (const auto& map_elem : map) {
        const auto& id = map_elem.first;
        auto iter = map_ref.find(id);
        if (iter == map_ref.end())
            return Result::error("Loop with id " + std::to_string(id) + " is not found in reference map");
        const auto& loop_info = map_elem.second;
        const auto& loop_info_ref = iter->second;
        PROPAGATE_ERROR("Loop with id " + std::to_string(id), compare_loop_info(loop_info, loop_info_ref));
    }
    return Result::ok();
}

LIRComparator::Result LIRComparator::compare_loop_info(const LoopInfoPtr& loop_info, const LoopInfoPtr& loop_info_ref) {
    COMPARE("Loop info type", loop_info->get_type_info(), loop_info_ref->get_type_info());
    COMPARE("Work amounts", loop_info->get_work_amount(), loop_info_ref->get_work_amount());
    COMPARE("Increments", loop_info->get_increment(), loop_info_ref->get_increment());

    PROPAGATE_ERROR("Input ports", compare_loop_ports(loop_info->get_input_ports(), loop_info_ref->get_input_ports()));
    PROPAGATE_ERROR("Output ports", compare_loop_ports(loop_info->get_output_ports(), loop_info_ref->get_output_ports()));
    if (ov::is_type<UnifiedLoopInfo>(loop_info)) {
        const auto unified_loop_info = ov::as_type_ptr<UnifiedLoopInfo>(loop_info);
        const auto unified_loop_info_ref = ov::as_type_ptr<UnifiedLoopInfo>(loop_info_ref);
        PROPAGATE_ERROR("Unified loop info", compare_unified_loop_info(unified_loop_info, unified_loop_info_ref));
    } else {
        const auto expanded_loop_info = ov::as_type_ptr<ExpandedLoopInfo>(loop_info);
        const auto expanded_loop_info_ref = ov::as_type_ptr<ExpandedLoopInfo>(loop_info_ref);
        PROPAGATE_ERROR("Expanded loop info", compare_expaned_loop_info(expanded_loop_info, expanded_loop_info_ref));
    }
    return Result::ok();
}

LIRComparator::Result LIRComparator::compare_unified_loop_info(const UnifiedLoopInfoPtr& loop_info,
                                                               const UnifiedLoopInfoPtr& loop_info_ref) {
    OPENVINO_ASSERT(loop_info && loop_info_ref, "compare_unified_loop_info got incorrect loop info");
    PROPAGATE_ERROR("Handlers", compare_handlers(loop_info->get_handlers(), loop_info_ref->get_handlers()));
    COMPARE("ptr increments", loop_info->get_ptr_increments(), loop_info_ref->get_ptr_increments());
    COMPARE("finalization offsets", loop_info->get_finalization_offsets(), loop_info_ref->get_finalization_offsets());
    COMPARE("data sizes", loop_info->get_data_sizes(), loop_info_ref->get_data_sizes());
    return Result::ok();
}

LIRComparator::Result LIRComparator::compare_expaned_loop_info(const ExpandedLoopInfoPtr& loop_info,
                                                               const ExpandedLoopInfoPtr& loop_info_ref) {
    OPENVINO_ASSERT(loop_info && loop_info_ref, "compare_expaned_loop_info got incorrect loop info");
    COMPARE("ptr increments", loop_info->get_ptr_increments(), loop_info_ref->get_ptr_increments());
    COMPARE("finalization offsets", loop_info->get_finalization_offsets(), loop_info_ref->get_finalization_offsets());
    COMPARE("data sizes", loop_info->get_data_sizes(), loop_info_ref->get_data_sizes());
    COMPARE("Loop iter type", loop_info->get_type(), loop_info_ref->get_type());
    const auto& unified_loop_info = loop_info->get_unified_loop_info();
    const auto& unified_loop_info_ref = loop_info_ref->get_unified_loop_info();
    PROPAGATE_ERROR("Unified loop info", compare_loop_info(unified_loop_info, unified_loop_info_ref));
    return Result::ok();
}

LIRComparator::Result LIRComparator::compare_loop_ports(const std::vector<LoopPort>& loop_ports,
                                                        const std::vector<LoopPort>& loop_ports_ref) {
    COMPARE("Loop ports size", loop_ports.size(), loop_ports_ref.size());
    for (size_t i = 0; i < loop_ports.size(); ++i) {
        const std::string prefix = "Loop port " + std::to_string(i) + ": ";
        COMPARE(prefix + "is_incremented", loop_ports[i].is_incremented, loop_ports_ref[i].is_incremented);
        COMPARE(prefix + "dim_idx", loop_ports[i].dim_idx, loop_ports_ref[i].dim_idx);
        PROPAGATE_ERROR(prefix + "expr_port", compare_expression_ports(*loop_ports[i].expr_port, *loop_ports_ref[i].expr_port));
    }
    return Result::ok();
}

LIRComparator::Result LIRComparator::compare_expression_ports(const ExpressionPort& expr_port,
                                                              const ExpressionPort& expr_port_ref) {
    const auto& port_node = expr_port.get_expr()->get_node();
    const auto& port_node_ref = expr_port_ref.get_expr()->get_node();
    COMPARE("port types", expr_port.get_type(), expr_port_ref.get_type());
    COMPARE("port index", expr_port.get_index(), expr_port_ref.get_index());
    COMPARE("port expression type", port_node->get_type_info(), port_node_ref->get_type_info());
    return Result::ok();
}

LIRComparator::Result LIRComparator::compare_port_connectors(const std::vector<PortConnectorPtr>& connectors,
                                                             const std::vector<PortConnectorPtr>& connectors_ref) {
    COMPARE("Port connectors size", connectors.size(), connectors_ref.size());
    for (size_t i = 0; i < connectors.size(); ++i) {
        const auto& connector = connectors[i];
        const auto& connector_ref = connectors_ref[i];
        PROPAGATE_ERROR("source port", compare_expression_ports(connector->get_source(), connector_ref->get_source()));

        const auto& consumers = connector->get_consumers();
        auto consumers_ref = connector_ref->get_consumers();
        COMPARE("consumers number", consumers.size(), consumers_ref.size());
        // Note: consumers are stored in std::set, and their order is not predefined
        for (const auto& consumer : consumers) {
            auto it = std::find_if(consumers_ref.begin(), consumers_ref.end(), [&consumer](const ExpressionPort& ref_port) {
                return compare_expression_ports(consumer, ref_port).valid;
            });
            if (it == consumers_ref.end())
                return Result::error("consumer was not found in consumers_ref");
            consumers_ref.erase(it);
        }
    }
    return Result::ok();
}

LIRComparator::Result LIRComparator::compare_handlers(const SpecificIterationHandlers& handlers,
                                                      const SpecificIterationHandlers& handlers_ref) {
    auto compare_pipelines = [](const ov::snippets::lowered::pass::PassPipeline& pipeline,
                                const ov::snippets::lowered::pass::PassPipeline& pipeline_ref) {
        const auto& passes = pipeline.get_passes();
        const auto& passes_ref = pipeline_ref.get_passes();
        COMPARE("Passes count", passes.size(), passes_ref.size());
        for (size_t i = 0; i < passes.size(); ++i) {
            const auto& pass = passes[i];
            const auto& pass_ref = passes_ref[i];
            if (pass->get_type_info() != pass_ref->get_type_info() || pass->merge(pass_ref) == nullptr) {
                return Result::error("Passes are not equal: " + std::string(pass->get_type_name()) + " and " +
                                     std::string(pass_ref->get_type_name()) +
                                     ". Pass names or parameters are not matched, or merge method is not overrided.");
            }
        }
        return Result::ok();
    };
    constexpr auto FIRST_ITER = SpecificLoopIterType::FIRST_ITER;
    constexpr auto MAIN_BODY = SpecificLoopIterType::MAIN_BODY;
    constexpr auto LAST_ITER = SpecificLoopIterType::LAST_ITER;
    PROPAGATE_ERROR("First iter passes", compare_pipelines(handlers.get_passes<FIRST_ITER>(), handlers_ref.get_passes<FIRST_ITER>()));
    PROPAGATE_ERROR("First iter passes", compare_pipelines(handlers.get_passes<MAIN_BODY>(), handlers_ref.get_passes<MAIN_BODY>()));
    PROPAGATE_ERROR("First iter passes", compare_pipelines(handlers.get_passes<LAST_ITER>(), handlers_ref.get_passes<LAST_ITER>()));
    return Result::ok();
}

#undef COMPARE
#undef PROPAGATE_ERROR

}  // namespace snippets
}  // namespace test
}  // namespace ov
