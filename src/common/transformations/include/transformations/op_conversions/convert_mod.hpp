// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <openvino/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>
#include <vector>

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ConvertMod;

}  // namespace pass
}  // namespace ov

class ov::pass::ConvertMod : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertMod", "0");
    ConvertMod();
};

namespace ngraph {
namespace pass {
using ov::pass::ConvertMod;
}  // namespace pass
}  // namespace ngraph
