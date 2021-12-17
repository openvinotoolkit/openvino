// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/cc/ngraph/itt.hpp>

#include "transformations/remove_converts.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset4.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pass/manager.hpp>

using NodeInput = ngraph::Input<ngraph::Node>;
using NodeOutput = ngraph::Output<ngraph::Node>;

namespace GNAPluginNS {
    NGRAPH_RTTI_DEFINITION(RemoveInputConvert, "RemoveInputConvert", 0);

    RemoveInputConvert::RemoveInputConvert() {
        MATCHER_SCOPE(RemoveInputConvert);

        const auto input = ngraph::pattern::wrap_type<ngraph::opset8::Parameter>();
        const auto convert = ngraph::pattern::wrap_type<ngraph::opset8::Convert>({input});

        ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
            const auto& pattern_map = m.get_pattern_value_map();
            auto node_it = pattern_map.find(convert);
            if (node_it == pattern_map.end())
                return false;
            auto convert_node = node_it->second.get_node_shared_ptr();
            auto input_node = convert_node->get_input_source_output(0).get_node_shared_ptr();

            // replace input precision with convert's one
            if (auto param = ov::as_type_ptr<ngraph::opset8::Parameter>(input_node)) {
                param->set_element_type(convert_node->output(0).get_tensor().get_element_type());
                param->validate_and_infer_types();
            }

            ngraph::replace_output_update_name(convert_node->output(0), convert_node->input_value(0));
            return true;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(convert, matcher_name);
        this->register_matcher(m, callback);
    }

    NGRAPH_RTTI_DEFINITION(RemoveOutputConvert, "RemoveOutputConvert", 0);
    RemoveOutputConvert::RemoveOutputConvert() {
        MATCHER_SCOPE(RemoveOutputConvert);

        auto output = ngraph::pattern::any_input();
        const auto convert = ngraph::pattern::wrap_type<ngraph::opset8::Convert>({output});
        const auto result = ngraph::pattern::wrap_type<ngraph::opset8::Result>({convert});

        ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
            const auto& pattern_map = m.get_pattern_value_map();
            auto convert_it = pattern_map.find(convert);
            if (convert_it == pattern_map.end())
                return false;
            auto convert_node = convert_it->second.get_node_shared_ptr();

            // the result presicion will be changed automaically
            ngraph::replace_output_update_name(convert_node->output(0), convert_node->input_value(0));
            return true;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(result, matcher_name);
        this->register_matcher(m, callback);
    }

} // namespace GNAPluginNS
