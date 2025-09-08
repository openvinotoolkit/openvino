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

class TRANSFORMATIONS_API ConvertNMS9ToNMSIEInternal;

}  // namespace pass
}  // namespace ov

class ov::pass::ConvertNMS9ToNMSIEInternal : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertNMS9ToNMSIEInternal");
    ConvertNMS9ToNMSIEInternal();
};
