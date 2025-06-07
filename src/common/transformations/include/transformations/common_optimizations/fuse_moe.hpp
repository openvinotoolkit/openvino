// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API FuseMOE;
class TRANSFORMATIONS_API FuseMOERouter;

}  // namespace pass
}  // namespace ov

class ov::pass::FuseMOE : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FuseMOE");
    FuseMOE();
};

class ov::pass::FuseMOERouter : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FuseMOERouter");
    FuseMOERouter();
};
