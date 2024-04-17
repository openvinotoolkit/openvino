// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/set_load_store_scalar.hpp"

#include "snippets/lowered/pass/insert_load_store.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/op/load.hpp"
#include "snippets/op/store.hpp"
#include "snippets/utils.hpp"
#include "snippets/itt.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

bool SetLoadStoreScalar::run(LinearIR& linear_ir, lowered::LinearIR::constExprIt begin, lowered::LinearIR::constExprIt end) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::SetLoadStoreScalar")
    bool modified = false;
    for (auto expr_it = begin; expr_it != end; ++expr_it) {
        const auto& expr = *expr_it;
        if (const auto load = ov::as_type_ptr<op::Load>(expr->get_node())) {
            const auto dim = utils::get_dim_idx(expr->get_input_port(0), 0);
            if (!utils::is_dynamic_value(dim) && dim == 1) {
                load->set_count(1);
                modified = true;
            }
        } else if (const auto store = ov::as_type_ptr<op::Store>(expr->get_node())) {
            const auto dim = utils::get_dim_idx(expr->get_output_port(0), 0);
            if (!utils::is_dynamic_value(dim) && dim == 1) {
                store->set_count(1);
                modified = true;
            }
        }
    }
    return modified;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
