// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "ngraph/ngraph_visibility.hpp"
#include "ngraph/node.hpp"
#include "ngraph/type.hpp"
#include "openvino/core/rt_info.hpp"

namespace ngraph {
using ov::copy_output_runtime_info;
using ov::copy_runtime_info;
}  // namespace ngraph

using ngraph::copy_runtime_info;
