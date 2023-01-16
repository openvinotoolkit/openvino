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

class TRANSFORMATIONS_API ConvertDivide;
class TRANSFORMATIONS_API ConvertDivideWithConstant;

}  // namespace pass
}  // namespace ov

class ov::pass::ConvertDivide : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertDivide", "0");
    ConvertDivide();
};

class ov::pass::ConvertDivideWithConstant : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertDivideWithConstant", "0");
    ConvertDivideWithConstant();
};

namespace ngraph {
namespace pass {
using ov::pass::ConvertDivide;
using ov::pass::ConvertDivideWithConstant;
}  // namespace pass
}  // namespace ngraph
