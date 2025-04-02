// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API MoeExpert2If;
class TRANSFORMATIONS_API FuseMoeExpert;

}  // namespace pass
}  // namespace ov

class ov::pass::MoeExpert2If : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("MoeExpert2If");
    MoeExpert2If();
};

class ov::pass::FuseMoeExpert : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FuseMoeExpert");
    FuseMoeExpert();
};
