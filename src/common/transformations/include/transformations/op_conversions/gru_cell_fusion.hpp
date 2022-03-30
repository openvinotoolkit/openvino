// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>

namespace ov {
namespace pass {

class TRANSFORMATIONS_API GRUCellFusion;

}  // namespace pass
}  // namespace ov


class ov::pass::GRUCellFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("GRUCellFusion", "0");
    GRUCellFusion();
};
