// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/cc/ngraph/itt.hpp>

#include "transformations/remove_single_input_concat.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pass/manager.hpp>

using NodeInput = ngraph::Input<ngraph::Node>;
using NodeOutput = ngraph::Output<ngraph::Node>;

namespace GNAPluginNS {
    RemoveSingleInputConcat::RemoveSingleInputConcat() {
        MATCHER_SCOPE(RemoveSingleInputConcat);

        auto is_required_node = [](const ngraph::Output<ngraph::Node>& value) {
            return value.get_node_shared_ptr()->get_input_size() == 1;
        };

        auto concat_operation = ngraph::pattern::wrap_type<ngraph::opset8::Concat>(is_required_node);

        ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
            const auto& pattern_map = m.get_pattern_value_map();
            auto concat_operation_it = pattern_map.find(concat_operation);
            if (concat_operation_it == pattern_map.end())
                return false;
            auto concat_operation_node = concat_operation_it->second.get_node_shared_ptr();

            NodeOutput prev_node_output = concat_operation_node->get_input_source_output(0);

            for (NodeInput child_input : concat_operation_node->get_output_target_inputs(0))
                child_input.replace_source_output(prev_node_output);

            return true;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(concat_operation, matcher_name);
        this->register_matcher(m, callback);
    }

} // namespace GNAPluginNS
