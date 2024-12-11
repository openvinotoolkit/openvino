// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ConvertGP9ToGPIEInternal;

}  // namespace pass
}  // namespace ov

class ov::pass::ConvertGP9ToGPIEInternal : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertGP9ToGPIEInternal");
    ConvertGP9ToGPIEInternal();
};
