// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset8.hpp>
#include <op_table.hpp>

using namespace std;
using namespace ngraph::opset8;

namespace ngraph {
namespace frontend {
namespace tf {
namespace op {

OutputVector TranslateRangeOp(const NodeContext& node) {
    auto start = node.get_ng_input(0);
    auto stop = node.get_ng_input(1);
    auto step = node.get_ng_input(2);
    auto out_type = node.get_attribute<ngraph::element::Type>("Tidx");

    auto range = make_shared<Range>(start, stop, step, out_type);
    range->set_friendly_name(node.get_name());
    return range->outputs();
}

}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ngraph
