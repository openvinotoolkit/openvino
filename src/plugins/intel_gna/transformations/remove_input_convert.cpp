// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/cc/ngraph/itt.hpp>

#include "transformations/remove_input_convert.hpp"

#include <memory>
#include <vector>

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
            auto input_it = pattern_map.find(convert);
            if (input_it == pattern_map.end())
                return false;
            auto input_node = input_it->second.get_node_shared_ptr();

            ngraph::replace_output_update_name(input_node->output(0), input_node->input_value(0));
            return true;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(convert, matcher_name);
        this->register_matcher(m, callback);
    }

    NGRAPH_RTTI_DEFINITION(RemoveOutputConvert, "RemoveOutputConvert", 0);
    RemoveOutputConvert::RemoveOutputConvert() {
        MATCHER_SCOPE(RemoveOutputConvert);

        const auto convert = ngraph::pattern::wrap_type<ngraph::opset8::Convert>();
        const auto output = ngraph::pattern::wrap_type<ngraph::opset8::Result>({convert});

        ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
            const auto& pattern_map = m.get_pattern_value_map();
            auto input_it = pattern_map.find(convert);
            if (input_it == pattern_map.end())
                return false;
            auto input_node = input_it->second.get_node_shared_ptr();

            ngraph::replace_output_update_name(input_node->output(0), input_node->input_value(0));
            return true;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(convert, matcher_name);
        this->register_matcher(m, callback);
    }

} // namespace GNAPluginNS
