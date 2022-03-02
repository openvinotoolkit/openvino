// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <snippets/itt.hpp>

#include "snippets/op/convert_saturation.hpp"
#include "snippets/pass/precision_propagation.hpp"
#include "ngraph_ops/type_relaxed.hpp"

#include <ngraph/rt_info.hpp>


ngraph::snippets::pass::PrecisionPropagation::PrecisionPropagation(const ov::element::Type exec_type) : exec_type(exec_type) { }

bool ngraph::snippets::pass::PrecisionPropagation::run_on_model(const std::shared_ptr<ov::Model> &m) {
    RUN_ON_FUNCTION_SCOPE(PrecisionPropagation);
    bool rewritten = false;
    for (auto& op : m->get_ordered_ops()) {
        if (auto node = std::dynamic_pointer_cast<ngraph::op::TypeRelaxedBase>(op)) {
            for (int i = 0; i < op->outputs().size(); i++) {
                node->set_overridden_output_type(exec_type, i);
                rewritten |= true;
            }
        } else {
            op->validate_and_infer_types();
        }
    }

    return rewritten;
}
