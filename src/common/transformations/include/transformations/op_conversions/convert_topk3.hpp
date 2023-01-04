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

class TRANSFORMATIONS_API ConvertTopK3;

}  // namespace pass
}  // namespace ov

class ov::pass::ConvertTopK3 : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertTopK3", "0");
    ConvertTopK3();
};

namespace ngraph {
namespace pass {
using ov::pass::ConvertTopK3;
}  // namespace pass
}  // namespace ngraph
