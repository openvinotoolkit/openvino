// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/pass/pass.hpp"
#include "openvino/pass/serialize.hpp"

namespace ngraph {
namespace pass {
using ov::pass::Serialize;
using ov::pass::StreamSerialize;
}  // namespace pass
}  // namespace ngraph
