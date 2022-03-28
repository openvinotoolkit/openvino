// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <snippets/itt.hpp>

#include "snippets/pass/precision_propagation.hpp"
#include "ngraph_ops/type_relaxed.hpp"

#include <ngraph/rt_info.hpp>


ngraph::snippets::pass::PrecisionPropagation::PrecisionPropagation(const ov::element::Type default_type) : default_type(default_type) { }

bool ngraph::snippets::pass::PrecisionPropagation::run_on_model(const std::shared_ptr<ov::Model> &m) {
    RUN_ON_FUNCTION_SCOPE(PrecisionPropagation);
    bool rewritten = false;
    for (auto& op : m->get_ordered_ops()) {
        // if there is convert with unsupported dst type we should insert new Convert with needed dst type after that
        // for correct math calculations
        if (auto convert = ov::as_type_ptr<ngraph::op::v0::Convert>(op)) {
            if (convert->get_destination_type() != default_type) {
               auto new_convert = std::make_shared<ov::op::v0::Convert>(convert, default_type);
                for (auto consumer : convert->output(0).get_target_inputs()) {
                    auto shared_consumer = consumer.get_node()->shared_from_this();
                    // After this Convert, there may be another Convert of the same kind (for example, it's special Convert before Result)
                    // So we shouldn't insert Convert with another destination type between these nodes
                    if (auto existing_convert = ov::as_type_ptr<ngraph::op::v0::Convert>(shared_consumer)) {
                        if (existing_convert->get_destination_type() == convert->get_destination_type())
                            continue;
                    } else if (!ov::is_type<ov::op::v0::Result>(shared_consumer) && shared_consumer != new_convert) {
                        consumer.replace_source_output(new_convert);
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
