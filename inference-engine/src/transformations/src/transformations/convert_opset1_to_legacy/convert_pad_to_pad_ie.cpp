// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/convert_opset1_to_legacy/convert_pad_to_pad_ie.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>

ngraph::pass::ConvertPadToLegacyMatcher::ConvertPadToLegacyMatcher() {
    ngraph::handler_callback callback = [](const std::shared_ptr<Node>& node) -> bool {
        auto pad = std::dynamic_pointer_cast<ngraph::opset1::Pad>(node);
        if (!pad) {
            return false;
        }

        auto pad_ie = std::make_shared<ngraph::op::PadIE>(pad);
        pad_ie->set_friendly_name(pad->get_friendly_name());
        ngraph::copy_runtime_info(pad, pad_ie);
        ngraph::replace_node(pad, pad_ie);
        return true;
    };

    this->register_matcher(callback);
}