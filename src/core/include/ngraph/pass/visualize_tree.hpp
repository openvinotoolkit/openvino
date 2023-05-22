// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#warning("The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")

#include <functional>
#include <set>
#include <sstream>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <utility>

#include "ngraph/pass/pass.hpp"
#include "openvino/pass/visualize_tree.hpp"

namespace ngraph {
namespace pass {
using ov::pass::VisualizeTree;
}  // namespace pass
}  // namespace ngraph
