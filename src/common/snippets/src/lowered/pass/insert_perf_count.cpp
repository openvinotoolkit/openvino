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

bool InsertPerfCount::run(LinearIR& linear_ir, lowered::LinearIR::constExprIt begin, lowered::LinearIR::constExprIt end) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::InsertPerfCount")
    if (m_boundary_op_names.empty()) {
        const auto& first_op_name = linear_ir.begin()->get()->get_node()->get_friendly_name();
        const auto& last_op_name = linear_ir.rbegin()->get()->get_node()->get_friendly_name();
        m_boundary_op_names.insert({first_op_name, last_op_name});
    }

    size_t seq_number = 0;
    for (auto expr_it = begin; expr_it != end; expr_it++) {
        const auto& op_name = expr_it->get()->get_node()->get_friendly_name();
        const auto& found = m_boundary_op_names.find(op_name);
        if (found != m_boundary_op_names.end()) {
            const auto perf_count_begin_pos = expr_it;
            auto perf_count_end_pos = expr_it;
            while (perf_count_end_pos->get()->get_node()->get_friendly_name() != found->second &&
                   perf_count_end_pos != linear_ir.cend()) {
                perf_count_end_pos++;
            }
            OPENVINO_ASSERT(perf_count_end_pos != linear_ir.cend(), "Failed to find requested op name to insert PerfCountEnd");
            const auto& perf_count_begin = std::make_shared<snippets::op::PerfCountBegin>();
            perf_count_begin->set_friendly_name(std::string("PerfCount_Begin_") + std::to_string(seq_number));
            const auto empty_inputs = std::vector<PortConnectorPtr>{};
            linear_ir.insert_node(perf_count_begin, empty_inputs, perf_count_begin_pos->get()->get_loop_ids(), false, perf_count_begin_pos);

            // Unique ConsoleDumper for each PerfCounter pair
            std::vector<std::shared_ptr<snippets::utils::Dumper>> dumpers;
            dumpers.push_back(std::make_shared<snippets::utils::ConsoleDumper>());

            const auto& perf_count_end = std::make_shared<snippets::op::PerfCountEnd>(perf_count_begin->output(0), dumpers);
            perf_count_end->set_friendly_name(std::string("PerfCount_End_") + std::to_string(seq_number));
            // linear_ir.insert has insert before behavior, need to increment perf_count_end_pos
            linear_ir.insert_node(perf_count_end, empty_inputs, perf_count_end_pos->get()->get_loop_ids(), false, next(perf_count_end_pos));
            seq_number++;
        }
    }
    return true;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
#endif  // SNIPPETS_DEBUG_CAPS
