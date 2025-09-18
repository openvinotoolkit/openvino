// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API FuseMOEExpert;
class TRANSFORMATIONS_API FuseMOERouter;
class TRANSFORMATIONS_API FuseMOE;

}  // namespace pass
}  // namespace ov

class ov::pass::FuseMOEExpert : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FuseMOE");
    FuseMOEExpert();
};

class ov::pass::FuseMOERouter : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FuseMOERouter");
    FuseMOERouter();
};

class ov::pass::FuseMOE : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("FuseMOE");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};
