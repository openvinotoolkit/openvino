// Copyright (C) 2018-2022 Intel Corporation
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
#include <ngraph/rt_info.hpp>

namespace GNAPluginNS {
    RemoveInputConvert::RemoveInputConvert() {
        MATCHER_SCOPE(RemoveInputConvert);

        const auto input = ngraph::pattern::wrap_type<ngraph::opset8::Parameter>(ngraph::pattern::type_matches_any(kSupportedInputTypesFrom));
        const auto convert = ngraph::pattern::wrap_type<ngraph::opset8::Convert>({input},
                                                                                 ngraph::pattern::type_matches_any(kSupportedInputTypesTo));

        ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
            const auto& pattern_map = m.get_pattern_value_map();

            auto convert_node = pattern_map.at(convert).get_node_shared_ptr();
            auto input_node = pattern_map.at(input).get_node_shared_ptr();

            // check the supported combinations
            auto from_type = input_node->get_element_type();
            auto to_type = convert_node->get_element_type();
            if (std::count(kSupportedInputConverts.begin(), kSupportedInputConverts.end(), std::make_pair(from_type, to_type)) == 0)
                return false;

            // replace input precision with convert's one
            if (auto param = ov::as_type_ptr<ngraph::opset8::Parameter>(input_node)) {
                param->set_element_type(to_type);
            }

            ngraph::replace_output_update_name(convert_node->output(0), convert_node->input_value(0));
            return true;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(convert, matcher_name);
        this->register_matcher(m, callback);
    }

    RemoveOutputConvert::RemoveOutputConvert() {
        MATCHER_SCOPE(RemoveOutputConvert);

        auto output = ngraph::pattern::any_input(ngraph::pattern::type_matches_any(kSupportedOutputTypesFrom));
        const auto convert = ngraph::pattern::wrap_type<ngraph::opset8::Convert>({output},
                                                                                 ngraph::pattern::type_matches_any(kSupportedOutputTypesTo));
        const auto result = ngraph::pattern::wrap_type<ngraph::opset8::Result>({convert},
                                                                               ngraph::pattern::type_matches_any(kSupportedOutputTypesTo));

        ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
            const auto& pattern_map = m.get_pattern_value_map();

            auto output_node = pattern_map.at(output).get_node_shared_ptr();
            auto convert_node = pattern_map.at(convert).get_node_shared_ptr();

            // check the supported combinations
            auto from_type = output_node->get_element_type();
            auto to_type = convert_node->get_element_type();
            if (std::count(kSupportedOutputConverts.begin(), kSupportedOutputConverts.end(), std::make_pair(from_type, to_type)) == 0) {
                return false;
            }

            // the result precision will be changed automatically
            ngraph::replace_output_update_name(convert_node->output(0), convert_node->input_value(0));
            return true;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(result, matcher_name);
        this->register_matcher(m, callback);
    }

} // namespace GNAPluginNS
