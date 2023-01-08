// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <openvino/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>
#include <vector>

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ConvertGELU;

}  // namespace pass
}  // namespace ov

class ov::pass::ConvertGELU : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertGELU", "0");
    ConvertGELU();
};

namespace ngraph {
namespace pass {
using ov::pass::ConvertGELU;
}  // namespace pass
}  // namespace ngraph
