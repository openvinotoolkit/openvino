// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <openvino/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>
#include <vector>

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ConvertNegative;

}  // namespace pass
}  // namespace ov

class ov::pass::ConvertNegative : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertNegative", "0");
    ConvertNegative();
};
