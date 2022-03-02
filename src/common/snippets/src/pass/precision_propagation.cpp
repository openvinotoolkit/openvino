// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <snippets/itt.hpp>

#include "snippets/pass/precision_propagation.hpp"
#include "ngraph_ops/type_relaxed.hpp"

#include <ngraph/rt_info.hpp>
#include <snippets/snippets_isa.hpp>


ngraph::snippets::pass::PrecisionPropagation::PrecisionPropagation(const ov::element::Type default_type) : default_type(default_type) { }

bool ngraph::snippets::pass::PrecisionPropagation::run_on_model(const std::shared_ptr<ov::Model> &m) {
    bool rewritten = false;
    for (auto& op : m->get_ops()) {
        // if there is convert with unsupported dst type we should insert new convert with needed dst type after that
        // for correct math calculations
        if (auto convert = ov::as_type_ptr<ngraph::op::v0::Convert>(op)) {
            if (convert->get_destination_type() != default_type) {
               auto re_convert = std::make_shared<ov::op::v0::Convert>(op, default_type);
                for (auto consumer : convert->output(0).get_target_inputs()) {
                    if (consumer.get_node()->shared_from_this() != convert) {
                        consumer.replace_source_output(convert);
                        rewritten |= true;
                    }
                }
            }
        } else if (auto node = std::dynamic_pointer_cast<ngraph::op::TypeRelaxedBase>(op)) {
            for (int i = 0; i < op->outputs().size(); i++) {
                node->set_overridden_output_type(default_type, i);
                rewritten |= true;
            }
        } else {
            op->validate_and_infer_types();
        }
    }

    return rewritten;
}
