// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_cloner.hpp"

namespace SubgraphsDumper {
const std::shared_ptr<ngraph::Node> clone_with_new_inputs(const std::shared_ptr<ngraph::Node> &node) {
    const auto node_type_name = node->get_type_name();
    if (ngraph::is_type<ngraph::op::v1::Convolution>(node) ) {
        case "Convolution":

    }
}
}