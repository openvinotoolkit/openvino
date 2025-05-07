// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#ifdef SNIPPETS_DEBUG_CAPS

#include "snippets/lowered/pass/insert_perf_count.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/itt.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

InsertPerfCount::InsertPerfCount(std::map<std::string, std::string> boundary_op_names)
    : RangedPass(), m_boundary_op_names(std::move(boundary_op_names)) {
}

// bool InsertPerfCount::run(LinearIR& linear_ir, lowered::LinearIR::constExprIt begin, lowered::LinearIR::constExprIt end) {
//     OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::InsertPerfCount")
//     if (m_boundary_op_names.empty()) {
//         const auto& first_op_name = linear_ir.begin()->get()->get_node()->get_friendly_name();
//         const auto& last_op_name = linear_ir.rbegin()->get()->get_node()->get_friendly_name();
//         m_boundary_op_names.insert({first_op_name, last_op_name});
//     }

//     size_t seq_number = 0;
//     for (auto expr_it = begin; expr_it != end; expr_it++) {
//         const auto& op_name = expr_it->get()->get_node()->get_friendly_name();
//         const auto& found = m_boundary_op_names.find(op_name);
//         if (found != m_boundary_op_names.end()) {
//             const auto perf_count_begin_pos = expr_it;
//             auto perf_count_end_pos = expr_it;
//             while (perf_count_end_pos->get()->get_node()->get_friendly_name() != found->second &&
//                    perf_count_end_pos != linear_ir.cend()) {
//                 perf_count_end_pos++;
//             }
//             OPENVINO_ASSERT(perf_count_end_pos != linear_ir.cend(), "Failed to find requested op name to insert PerfCountEnd");
//             const auto& perf_count_begin = std::make_shared<snippets::op::PerfCountBegin>();
//             perf_count_begin->set_friendly_name(std::string("PerfCount_Begin_") + std::to_string(seq_number));
//             const auto empty_inputs = std::vector<PortConnectorPtr>{};
//             linear_ir.insert_node(perf_count_begin, empty_inputs, perf_count_begin_pos->get()->get_loop_ids(), false, perf_count_begin_pos);

//             // Unique ConsoleDumper for each PerfCounter pair
//             std::vector<std::shared_ptr<snippets::utils::Dumper>> dumpers;
//             dumpers.push_back(std::make_shared<snippets::utils::ConsoleDumper>());

//             const auto& perf_count_end = std::make_shared<snippets::op::PerfCountEnd>(perf_count_begin->output(0), dumpers);
//             perf_count_end->set_friendly_name(std::string("PerfCount_End_") + std::to_string(seq_number));
//             // linear_ir.insert has insert before behavior, need to increment perf_count_end_pos
//             linear_ir.insert_node(perf_count_end, empty_inputs, perf_count_end_pos->get()->get_loop_ids(), false, next(perf_count_end_pos));
//             seq_number++;
//         }
//     }
//     return true;
// }

bool InsertPerfCount::run(LinearIR& linear_ir, lowered::LinearIR::constExprIt begin, lowered::LinearIR::constExprIt end) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::InsertPerfCount")

    auto is_parameter = [](const std::shared_ptr<ov::Node>& node) {
        return ov::is_type<ov::op::v0::Parameter>(node);
    };
    auto is_result = [](const std::shared_ptr<ov::Node>& node) {
        return ov::is_type<ov::op::v0::Result>(node);
    };

    // mark perf_count_begin and perf_count_end position
    auto perf_count_begin_pos = linear_ir.cbegin();
    auto perf_count_end_pos = perf_count_begin_pos;
    bool first_result_marked = false;
    for (auto expr_it = linear_ir.cbegin(); expr_it != linear_ir.cend(); expr_it++) {
        const auto expr = *expr_it;
        const auto& node = expr->get_node();
        if (is_parameter(node))
            perf_count_begin_pos = expr_it;

        if (is_result(node) && !first_result_marked) {
            perf_count_end_pos = expr_it;
            first_result_marked = true;
        }
    }

    // insert perf_count_begin after last parameter
    // linear_ir.insert has insert before behavior, need move to next.
    const auto& perf_count_begin = std::make_shared<snippets::op::PerfCountBegin>();
    perf_count_begin->set_friendly_name(std::string("PerfCount_Begin"));
    const auto empty_inputs = std::vector<PortConnectorPtr>{};
    linear_ir.insert_node(perf_count_begin, empty_inputs, perf_count_begin_pos->get()->get_loop_ids(), false, next(perf_count_begin_pos));

    // Unique ConsoleDumper for each PerfCounter pair
    std::vector<std::shared_ptr<snippets::utils::Dumper>> dumpers;
    // Add CSV dumper if path is provided
    // Collect brgemm parameters
    std::string params;
    for (auto expr_it = begin; expr_it != end; expr_it++) {
        const auto& brgemm_expr = *expr_it;
        const auto brgemm = ov::as_type_ptr<ov::snippets::op::Brgemm>(brgemm_expr->get_node());
        if (!brgemm)
            continue;
        // Collect brgemm parameters
        params = collect_params(brgemm_expr, linear_ir);
        break;
    }

    auto csv_path = linear_ir.get_config().debug_config->dumpParams.csv_path;
    if (!csv_path.empty()) {
        dumpers.push_back(std::make_shared<snippets::utils::CSVDumper>(csv_path));
    }
    dumpers.push_back(std::make_shared<snippets::utils::ConsoleDumper>());
    // insert perf_count_end before first result
    const auto& perf_count_end = std::make_shared<snippets::op::PerfCountEnd>(perf_count_begin->output(0), dumpers, params);
    perf_count_end->set_friendly_name(std::string("PerfCount_End"));
    linear_ir.insert_node(perf_count_end, empty_inputs, perf_count_end_pos->get()->get_loop_ids(), false, perf_count_end_pos);
    return true;
}

std::string InsertPerfCount::collect_params(const ov::snippets::lowered::ExpressionPtr& brgemm_expr,
                                                   const snippets::lowered::LinearIR& linear_ir) {
    const auto brgemm = ov::as_type_ptr<ov::snippets::op::Brgemm>(brgemm_expr->get_node());
    OPENVINO_ASSERT(brgemm, "Brgemm is nullptr!");
    std::stringstream ss;
    ss << "MatMul_subgraph" << ',';
    ss << brgemm_expr->get_node()->get_friendly_name() << ',';
    for (size_t i = 0; i < brgemm->get_input_size(); ++i) {
        ss << brgemm->get_input_element_type(i);
        if (i != brgemm->get_input_size() - 1) {
            ss << ';';
        }
    }
    ss << ',';
    for (size_t i = 0; i < brgemm->get_output_size(); ++i) {
        ss << brgemm->get_output_element_type(i);
        if (i != brgemm->get_output_size() - 1) {
            ss << ';';
        }
    }
    ss << ',';
    for (size_t i = 0; i < brgemm->inputs().size(); ++i) {
        const auto& port_desc = brgemm_expr->get_input_port_descriptor(i);
        const auto& shape = ov::snippets::utils::get_planar_vdims(port_desc->get_shape(), port_desc->get_layout());
        ss << utils::tensor2str(shape, " ");
        ss << ';';
    }
    ss.seekp(-1, ss.cur);
    ss << ',';
    for (size_t i = 0; i < brgemm->outputs().size(); ++i) {
        const auto& port_desc = brgemm_expr->get_output_port_descriptor(i);
        const auto& shape = ov::snippets::utils::get_preordered_vdims(port_desc->get_shape(), port_desc->get_layout());
        ss << utils::tensor2str(shape, " ");
        ss << ';';
    }
    ss.seekp(-1, ss.cur);
    ss << ',';
    for (size_t i = 0; i < brgemm->inputs().size(); ++i) {
        const auto& port_desc = brgemm_expr->get_input_port_descriptor(i);
        ss << utils::tensor2str(port_desc->get_layout(), " ");
        ss << ';';
    }
    ss << ',';
    for (size_t i = 0; i < brgemm->outputs().size(); ++i) {
        const auto& port_desc = brgemm_expr->get_output_port_descriptor(i);
        ss << utils::tensor2str(port_desc->get_layout(), " ");
        ss << ';';
    }
    ss << ',';

    const auto& in_0_desc = brgemm_expr->get_input_port_descriptor(0);
    const auto& in_1_desc = brgemm_expr->get_input_port_descriptor(1);
    const auto& out_desc = brgemm_expr->get_output_port_descriptor(0);

    const auto& in_0_planar_dims =
        ov::snippets::utils::get_planar_vdims(in_0_desc->get_shape(), in_0_desc->get_layout());
    const auto& in_1_planar_dims =
        ov::snippets::utils::get_planar_vdims(in_1_desc->get_shape(), in_1_desc->get_layout());
    const auto& out_preordered_dims =
        ov::snippets::utils::get_preordered_vdims(out_desc->get_shape(), out_desc->get_layout());

    const auto& m = *++out_preordered_dims.rbegin();
    const auto& n = *out_preordered_dims.rbegin();
    const auto& k0 = *in_0_planar_dims.rbegin();
    const auto& k1 = *++in_1_planar_dims.rbegin();
    size_t k = 0;
    OPENVINO_ASSERT(utils::merge_dynamic_dim(k, k0, k1),
                    "Brgemm input descriptors have incompatible K dimension value.");
    ss << static_cast<int64_t>(m) << ',' << static_cast<int64_t>(n) << ',' << static_cast<int64_t>(k) << ',';

    size_t m_block = in_0_desc->get_subtensor().front();
    size_t n_block = in_1_desc->get_subtensor().back();
    size_t k_block = in_0_desc->get_subtensor().back();

    auto append_block_info = [&](size_t block) {
        if (block == utils::get_full_dim_value()) {
            ss << "FULL_DIM";
        } else if (block == utils::get_dynamic_value<size_t>()) {
            ss << "?";
        } else {
            ss << block;
        }
        ss << ',';
    };

    append_block_info(m_block);
    append_block_info(n_block);
    append_block_info(k_block);
    return ss.str();
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
#endif  // SNIPPETS_DEBUG_CAPS
