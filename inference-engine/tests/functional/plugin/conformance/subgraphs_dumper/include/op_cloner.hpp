// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/ngraph.hpp>

namespace SubgraphsDumper {
template<typename OPType>
const std::shared_ptr<ngraph::Node> clone_with_new_inputs(const std::shared_ptr<OPType> &node);

using cloners_map_type = std::map<ngraph::NodeTypeInfo,
        std::function<const std::shared_ptr<ngraph::Node>(const std::shared_ptr<ngraph::Node> &node)>>;

static cloners_map_type cloners_map{
#define NGRAPH_OP(NAME, NAMESPACE) {NAMESPACE::NAME::type_info, clone_with_new_inputs<NAMESPACE::NAME>},

#include <ngraph/opsets/opset6_tbl.hpp>

#undef NGRAPH_OP
};
}  // namespace SubgraphsDumper
