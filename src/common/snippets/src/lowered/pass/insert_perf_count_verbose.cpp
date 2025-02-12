// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef SNIPPETS_DEBUG_CAPS
#include "snippets/lowered/pass/insert_perf_count_verbose.hpp"

#include "snippets/itt.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/lowered/pass/pass.hpp"
#include "snippets/lowered/pass/propagate_subtensors.hpp"
#include "snippets/lowered/pass/iter_handler.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/utils/utils.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

bool InsertPerfCountVerbose::run(snippets::lowered::LinearIR& linear_ir,
                                 snippets::lowered::LinearIR::constExprIt begin,
                                 snippets::lowered::LinearIR::constExprIt end) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::InsertPerfCountVerbose")

    static size_t seq_number = 0;
    bool modified = false;
    auto csv_path = linear_ir.get_config().debug_config->dumpParams.csv_path;

    std::vector<std::shared_ptr<snippets::utils::Dumper>> dumpers;
    dumpers.push_back(std::make_shared<snippets::utils::ConsoleDumper>());
    // Add CSV dumper if path is provided
    if (!linear_ir.get_config().debug_config->dumpParams.csv_path.empty()) {
        dumpers.push_back(std::make_shared<snippets::utils::CSVDumper>(csv_path));
    }

    for (auto expr_it = begin; expr_it != end; expr_it++) {
        const auto& brgemm_expr = *expr_it;
        const auto brgemm = ov::as_type_ptr<ov::snippets::op::Brgemm>(brgemm_expr->get_node());
        if (!brgemm)
            continue;
        // Collect brgemm parameters
        auto params = collect_params(brgemm_expr, linear_ir);

        const auto& perf_count_begin = std::make_shared<snippets::op::PerfCountBegin>();
        perf_count_begin->set_friendly_name(std::string("PerfCountVerbose_Begin_") + std::to_string(seq_number));
        const auto empty_inputs = std::vector<PortConnectorPtr>{};
        linear_ir.insert_node(perf_count_begin, empty_inputs, expr_it->get()->get_loop_ids(), false, expr_it);

        const auto& perf_count_end = std::make_shared<snippets::op::PerfCountEnd>(perf_count_begin->output(0), dumpers, params);
        perf_count_end->set_friendly_name(std::string("PerfCountVerbose_End_") + std::to_string(seq_number));

        linear_ir.insert_node(perf_count_end, empty_inputs, expr_it->get()->get_loop_ids(), false, next(expr_it));
        seq_number++;
        modified = true;
    }
    return modified;
}

std::string InsertPerfCountVerbose::collect_params(const ov::snippets::lowered::ExpressionPtr& brgemm_expr,
                                                   const snippets::lowered::LinearIR& linear_ir) {
    const auto brgemm = ov::as_type_ptr<ov::snippets::op::Brgemm>(brgemm_expr->get_node());
    OPENVINO_ASSERT(brgemm, "Brgemm is nullptr!");
    std::stringstream ss;
    ss << m_subgraph_name << ',';
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
    size_t k_block = out_desc->get_subtensor().back();

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

#endif // SNIPPETS_DEBUG_CAPS
