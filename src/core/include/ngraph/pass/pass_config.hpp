// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#warning("The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")

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
