// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>

namespace ov {
namespace intel_gpu {

class ConvertShapeOf1To3 : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertShapeOf1To3", "0");
    ConvertShapeOf1To3();
};

}  // namespace intel_gpu
}  // namespace ov
