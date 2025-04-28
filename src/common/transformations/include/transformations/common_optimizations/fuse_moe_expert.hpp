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
class TRANSFORMATIONS_API FuseMoeExpert2;
class TRANSFORMATIONS_API FuseMoeExpertPlain;
class TRANSFORMATIONS_API FuseMoeExpertOneHot;
class TRANSFORMATIONS_API FuseMoeExpertSoftTopK;

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

class ov::pass::FuseMoeExpert2 : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FuseMoeExpert2");
    FuseMoeExpert2();
};

class ov::pass::FuseMoeExpertPlain : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FuseMoeExpert3");
    FuseMoeExpertPlain();
};

class ov::pass::FuseMoeExpertOneHot : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FuseMoeExpertOneHot");
    FuseMoeExpertOneHot();
};

class ov::pass::FuseMoeExpertSoftTopK : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FuseMoeExpertSoftTopK");
    FuseMoeExpertSoftTopK();
};
