// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/rt_info/reverse_input_channels.hpp"

#include "ngraph/node.hpp"

bool ov::is_reverse_input_channels(const std::shared_ptr<ngraph::Node>& node) {
    return node->get_rt_info().count(ReverseInputChannels::get_type_info_static());
}

void ov::set_is_reverse_input_channels(std::shared_ptr<ngraph::Node> node) {
    node->get_rt_info().emplace(ReverseInputChannels::get_type_info_static(), ReverseInputChannels{});
}