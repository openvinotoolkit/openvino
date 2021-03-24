// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <memory>
#include <sstream>

#include "ngraph/node.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/type/element_type.hpp"

using namespace ngraph;
using namespace std;

op::Op::Op(const OutputVector& args)
    : Node(args)
{
}
