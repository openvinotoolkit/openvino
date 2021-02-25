// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/ngraph.hpp>

namespace SubgraphsDumper {
template <typename opType>
const std::shared_ptr<ngraph::Node> clone_node(const std::shared_ptr<ngraph::Node> &node);

using cloners_map_type = std::map<ngraph::NodeTypeInfo,
        std::function<const std::shared_ptr<ngraph::Node>(const std::shared_ptr<ngraph::Node> &node)>>;

static cloners_map_type cloners_map{
#define NGRAPH_OP(NAME, NAMESPACE) {NAMESPACE::NAME::type_info, clone_node<NAMESPACE::NAME>},

#include <ngraph/opsets/opset1_tbl.hpp>
#include <ngraph/opsets/opset2_tbl.hpp>
#include <ngraph/opsets/opset3_tbl.hpp>
#include <ngraph/opsets/opset4_tbl.hpp>
#include <ngraph/opsets/opset5_tbl.hpp>
#include <ngraph/opsets/opset6_tbl.hpp>

#undef NGRAPH_OP
};
}  // namespace SubgraphsDumper
