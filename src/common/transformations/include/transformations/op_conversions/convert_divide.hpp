// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

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
