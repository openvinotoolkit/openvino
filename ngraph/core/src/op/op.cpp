// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/op.hpp"

#include <algorithm>
#include <memory>
#include <sstream>

#include "ngraph/node.hpp"
#include "ngraph/type/element_type.hpp"

using namespace ngraph;
using namespace std;

op::Op::Op(const OutputVector& args) : Node(args) {}
const ::ov::Node::type_info_t op::Op::type_info{"Op", 0, "util"};
const ::ov::Node::type_info_t& op::Op::get_type_info() const {
    static const ::ov::Node::type_info_t info{"Op", 0, "util"};
    return info;
}
