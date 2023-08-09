// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <openvino/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>

namespace ov {
namespace pass {
class TRANSFORMATIONS_API RPE_Optimization;
}  // namespace pass
}  // namespace ov

class ov::pass::RPE_Optimization : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("RPE_Optimization", "0");
    RPE_Optimization();
};
