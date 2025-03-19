// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/op.hpp"

#include <algorithm>
#include <memory>
#include <sstream>

#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"

ov::op::Op::Op(const ov::OutputVector& args) : Node(args) {}
