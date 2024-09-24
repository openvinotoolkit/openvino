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

class TRANSFORMATIONS_API ConvertBroadcast3;

}  // namespace pass
}  // namespace ov

class ov::pass::ConvertBroadcast3 : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertBroadcast3", "0");
    ConvertBroadcast3();
};
