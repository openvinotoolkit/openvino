// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>

namespace ov {
namespace pass {

class TRANSFORMATIONS_API EnableShapeOfConstantFolding : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("EnableShapeOfConstantFolding", "0");
    EnableShapeOfConstantFolding();
};

}  // namespace pass
}  // namespace ov
