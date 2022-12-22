// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/convert_to_snippets_opset.hpp"

#include <ngraph/rt_info.hpp>
#include "snippets/itt.hpp"
#include "snippets/op/add.hpp"
#include "snippets/op/convert_saturation.hpp"

#ifdef CPU_DEBUG_CAPS_SNIPPETS
#include "ngraph/pass/visualize_tree.hpp"
#endif

#define REPLACE_AND_CONTINUE(instance, type_from, type_to)                   \
    if (is_type<type_from>(instance)) {\
    auto new_instance = std::make_shared<type_to>(*as_type_ptr<type_from>(instance));\
    replace_node(instance, new_instance);\
    copy_runtime_info(instance, new_instance);\
    was_updated = true;\
    continue;\
}\

void REPLACE_AND_CONTINUE_DEBUG(const std::shared_ptr<ngraph::Node>& instance) {
    if (ngraph::is_type<ngraph::opset1::Add>(instance)) {
        std::vector<ov::element::Type> output_types;
        // std::all
        for (const auto& output : instance->outputs()) {
            output_types.push_back(output.get_element_type());
        }
        auto new_instance = std::make_shared<ngraph::snippets::op::Add>(*ngraph::as_type_ptr<ngraph::opset1::Add>(instance));
        replace_node(instance, new_instance);
        copy_runtime_info(instance, new_instance);

        new_instance->validate_and_infer_types();
        for (size_t i = 0; i < new_instance->get_output_size(); ++i) {
            const auto& output = new_instance->output(i);
            if (output.get_element_type() != output_types[i]) {
                for (auto& input : output.get_target_inputs()) {
                    auto convert = std::make_shared<ngraph::snippets::op::ConvertSaturation>(output, output_types[i]);
                    input.replace_source_output(convert->output(0));
                }
            }
        }
        return;
    }
}

bool ngraph::snippets::pass::ConvertToSnippetsOpset::run_on_model(const std::shared_ptr<ov::Model>& f) {
    RUN_ON_MODEL_SCOPE(ConvertToSnippetsOpset);
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::op::ConvertToSnippetsOpset")

    bool was_updated = false;
    const auto& ops = f->get_ordered_ops();
    for (const auto& op : f->get_ordered_ops()) {
        // add comment: why we need add to operation here
        //REPLACE_AND_CONTINUE(op, ngraph::opset1::Add, ngraph::snippets::op::Add)

        REPLACE_AND_CONTINUE_DEBUG(op);
    }

    return was_updated;
}
