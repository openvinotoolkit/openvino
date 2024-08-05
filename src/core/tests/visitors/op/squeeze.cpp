// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/squeeze.hpp"

#include "unary_ops.hpp"

using Types = ::testing::Types<UnaryOperatorType<ov::op::v0::Squeeze, ov::element::f32>,
                               UnaryOperatorType<ov::op::v0::Squeeze, ov::element::f16>>;
