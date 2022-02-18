// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/adopt_convert.hpp"

#include "remarks.hpp"
#include <snippets/itt.hpp>

#include "snippets/op/subgraph.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/op/loop.hpp>
#include "transformations/utils/utils.hpp"

#include <memory>
#include <vector>
#include <cassert>
#include <queue>
#include <string>
#include <numeric>
#include <climits>

NGRAPH_RTTI_DEFINITION(ngraph::snippets::pass::AdoptConvert, "Snippets::AdoptConvert", 0);

namespace ngraph {
namespace snippets {
namespace pass {

AdoptConvert::AdoptConvert(const std::vector<ov::element::Type> supported_output_types) {
    MATCHER_SCOPE(AdoptConvert);
    // TODO: extend pattern to use supportedTypes
    auto convertWrapper = ngraph::pattern::wrap_type<opset1::Convert>();

    ngraph::graph_rewrite_callback callback = [&,supported_output_types](ngraph::pattern::Matcher& m) -> bool {
        OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::AdoptConvert_callback")
        // TODO: move to pattern
        auto convert = m.get_match_root();
        const auto convet_output_type = convert->get_output_element_type(0);
        if (std::any_of(
            supported_output_types.begin(),
            supported_output_types.end(),
            [&convet_output_type](const auto& type) { return convet_output_type == type; })) {
            return false;
        }

        bool precisionIsRestored = true;
        for (const auto& output : convert->outputs()) {
            for (const auto& childInput : output.get_target_inputs()) {
                const auto& childNode = childInput.get_node();
                if (!ov::is_type<opset1::Convert>(childNode)) {
                    precisionIsRestored = false;
                    break;
                };

                const auto output_type = childNode->get_output_element_type(0);
                if (std::all_of(
                    supported_output_types.begin(),
                    supported_output_types.end(),
                    [&output_type](const auto& type) { return output_type != type; })) {
                    precisionIsRestored = false;
                    break;
                }
            }
        }

        if (precisionIsRestored) {
            return true;
        }

        if (transformation_callback(convert)) {
            return false;
        }

        // TODO: reconnect existing instead to new creation
        const auto originalConvert = std::make_shared<ngraph::opset1::Convert>(
            convert->get_input_node_shared_ptr(0),
            convert->get_output_element_type(0));
        originalConvert->set_friendly_name(m.get_match_root()->get_friendly_name());

        const auto newConvert = std::make_shared<ngraph::opset1::Convert>(
            originalConvert,
            convert->get_input_element_type(0));
        newConvert->set_friendly_name(m.get_match_root()->get_friendly_name() + "/adoption");

        ngraph::copy_runtime_info(convert, {originalConvert, newConvert});
        replace_node(convert, newConvert);
        return true;
    };
    auto matcher = std::make_shared<ngraph::pattern::Matcher>(convertWrapper);
    register_matcher(matcher, callback);
}

} // namespace pass
} // namespace snippets
} // namespace ngraph