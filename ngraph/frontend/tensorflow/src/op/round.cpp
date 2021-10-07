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

OutputVector TranslateRoundOp(const NodeContext& node) {
    auto input = node.get_ng_input(0);
    // using default round mode "half_to_even" in openvino,
    // as TF has only that mode
    auto round_mode = Round::RoundMode::HALF_TO_EVEN;
    auto round = make_shared<Round>(input, round_mode);
    round->set_friendly_name(node.get_name());
    return round->outputs();
}
}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ngraph
