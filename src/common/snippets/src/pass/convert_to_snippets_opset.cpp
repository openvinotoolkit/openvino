// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/convert_to_snippets_opset.hpp"

#include <ngraph/rt_info.hpp>
#include "snippets/itt.hpp"
#include "snippets/op/convert_saturation.hpp"

template<typename type_from, typename type_to>
bool replace(const std::shared_ptr<ngraph::Node>& instance) {
    if (ngraph::is_type<type_from>(instance)) {
        std::vector<ov::element::Type> output_types;
        for (const auto& output : instance->outputs()) {
            output_types.push_back(output.get_element_type());
        }
        auto new_instance = std::make_shared<type_to>(*ngraph::as_type_ptr<type_from>(instance));
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
        return true;
    }
    return false;
}

bool ngraph::snippets::pass::ConvertToSnippetsOpset::run_on_model(const std::shared_ptr<ov::Model>& f) {
    RUN_ON_MODEL_SCOPE(ConvertToSnippetsOpset);
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::op::ConvertToSnippetsOpset")

    bool was_updated = false;
    const auto& ops = f->get_ordered_ops();
    for (const auto& op : f->get_ordered_ops()) {
        // TODO: example how to use
        //if (replace<ngraph::opset1::Add, ngraph::snippets::op::Add>(op)) {
        //    continue;
        //}
    }

    return was_updated;
}
