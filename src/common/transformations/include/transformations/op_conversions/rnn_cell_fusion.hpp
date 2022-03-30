// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>

namespace ov {
namespace pass {

class TRANSFORMATIONS_API RNNCellFusion;

}  // namespace pass
}  // namespace ov


class ov::pass::RNNCellFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("RNNCellFusion", "0");
    RNNCellFusion();
};
