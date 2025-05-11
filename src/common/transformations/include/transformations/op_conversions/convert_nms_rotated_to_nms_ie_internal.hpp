// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ConvertNMSRotatedToNMSIEInternal;

}  // namespace pass
}  // namespace ov

class ov::pass::ConvertNMSRotatedToNMSIEInternal : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertNMSRotatedToNMSIEInternal");
    ConvertNMSRotatedToNMSIEInternal();
};
