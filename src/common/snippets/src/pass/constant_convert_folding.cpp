// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <snippets/itt.hpp>

#include "snippets/pass/constant_convert_folding.hpp"
#include "ngraph/pass/constant_folding.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

const auto friendly_name_from = [](const ov::Node& node, const size_t output_count, const size_t idx) {
    constexpr auto single_output = static_cast<size_t>(1);
    return single_output == output_count ? node.get_friendly_name() : node.get_friendly_name() + "." + std::to_string(idx);
};

ngraph::snippets::pass::ConstantConvertFolding::ConstantConvertFolding() {
    MATCHER_SCOPE(ConstantConvertFolding);
    auto constant_pattern = ngraph::pattern::wrap_type<ngraph::opset1::Constant>();
    auto convert_pattern = ngraph::pattern::wrap_type<ngraph::opset1::Convert>({constant_pattern});

    register_matcher(std::make_shared<ngraph::pattern::Matcher>(convert_pattern, matcher_name),
        [=](ngraph::pattern::Matcher &m) {
            OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::op::ConstantConvertFolding")
            auto root = m.get_match_root();

            const auto &pm = m.get_pattern_value_map();
            const auto constant = pm.at(constant_pattern).get_node_shared_ptr();
            const auto convert = pm.at(convert_pattern).get_node_shared_ptr();

            bool rewritten = false;
            OutputVector replacements(convert->get_output_size());
            if (convert->constant_fold(replacements, convert->input_values())) {
                const bool constant_folding_is_disabled = convert->get_rt_info().count(ov::pass::DisableConstantFolding::get_type_info_static());
                OPENVINO_ASSERT(!constant_folding_is_disabled,
                                "Node folded but constant folding disabled. Check constant_fold implementation for ",
                                convert);
                OPENVINO_ASSERT(replacements.size() == convert->get_output_size(),
                                "constant_fold_default returned incorrect number of replacements for ",
                                convert);

                for (size_t i = 0; i < replacements.size(); ++i) {
                    auto node_output = convert->output(i);
                    auto replacement = replacements.at(i);
                    if (replacement.get_node_shared_ptr() && (node_output != replacement)) {
                        replacement.get_node()->set_friendly_name(friendly_name_from(*convert, replacements.size(), i));

                        node_output.replace(replacement);
                        // Propagate runtime info attributes to replacement consumer nodes
                        for (auto& input : replacement.get_target_inputs()) {
                            auto consumer = input.get_node()->shared_from_this();
                            copy_runtime_info({convert, consumer}, consumer);
                        }

                        rewritten = true;
                    }
                }
            }
            return rewritten;
        });
}