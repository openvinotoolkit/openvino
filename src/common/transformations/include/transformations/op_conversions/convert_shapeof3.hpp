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

class TRANSFORMATIONS_API ConvertShapeOf3;

}  // namespace pass
}  // namespace ov

class ov::pass::ConvertShapeOf3 : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertShapeOf3", "0");
    ConvertShapeOf3();
};
