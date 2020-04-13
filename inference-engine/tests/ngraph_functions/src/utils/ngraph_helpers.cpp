// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>

#include <ngraph/opsets/opset1.hpp>

namespace ngraph {
namespace helpers {

ngraph::OutputVector convert2OutputVector(const std::vector<std::shared_ptr<ngraph::Node>> &nodes) {
    ngraph::OutputVector outs;
    std::for_each(nodes.begin(), nodes.end(), [&outs](const std::shared_ptr<ngraph::Node> &n) {
        for (const auto &out_p : n->outputs()) {
            outs.push_back(out_p);
        }
    });
    return outs;
}

template<class opType>
ngraph::NodeVector castOps2Nodes(const std::vector<std::shared_ptr<opType>> &ops) {
    ngraph::NodeVector nodes;
    for (const auto &op : ops) {
        nodes.push_back(std::dynamic_pointer_cast<ngraph::Node>(op));
    }
    return nodes;
}

}  // namespace helpers
}  // namespace ngraph
