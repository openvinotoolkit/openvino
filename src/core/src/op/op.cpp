// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/op.hpp"

#include <algorithm>
#include <memory>
#include <sstream>

#include "ngraph/node.hpp"
#include "ngraph/type/element_type.hpp"

using namespace std;

ov::op::Op::Op(const ov::OutputVector& args) : Node(args) {}
