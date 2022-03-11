// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <list>
#include <memory>
#include <vector>

#include "ngraph/deprecated.hpp"
#include "ngraph/function.hpp"
#include "ngraph/node.hpp"
#include "ngraph/util.hpp"
#include "openvino/pass/pass_config.hpp"

namespace ngraph {
namespace pass {
using ov::pass::param_callback;
using ov::pass::param_callback_map;
using ov::pass::PassConfig;
}  // namespace pass
}  // namespace ngraph
