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

OutputVector TranslatePackOp(const NodeContext& node) {
    auto axis = node.get_attribute<int32_t>("axis");
    auto axis_const = make_shared<Constant>(element::i64, Shape{}, axis);

    OutputVector concat_inputs;
    for (size_t i = 0; i < node.get_ng_input_size(); ++i) {
        auto in = node.get_ng_input(i);
        concat_inputs.push_back(make_shared<Unsqueeze>(in, axis_const));
    }

    auto concat = make_shared<Concat>(concat_inputs, axis);
    concat->set_friendly_name(node.get_name());
    return concat->outputs();
}
}
}
}
}