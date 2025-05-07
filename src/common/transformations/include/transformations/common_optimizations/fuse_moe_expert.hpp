// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API FuseMoeExpert;
class TRANSFORMATIONS_API FuseMoeExpertRouter;

}  // namespace pass
}  // namespace ov

class ov::pass::FuseMoeExpert : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FuseMoeExpert");
    FuseMoeExpert();
};

class ov::pass::FuseMoeExpertRouter : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FuseMoeExpertRouter");
    FuseMoeExpertRouter();
};
