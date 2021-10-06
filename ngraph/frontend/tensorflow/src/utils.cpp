// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"

void ngraph::frontend::tf::SetTracingInfo(const std::string& op_name, const ngraph::Output<ngraph::Node> ng_node) {
    auto node = ng_node.get_node_shared_ptr();
    node->set_friendly_name(op_name);
    node->add_provenance_tag(op_name);
}
