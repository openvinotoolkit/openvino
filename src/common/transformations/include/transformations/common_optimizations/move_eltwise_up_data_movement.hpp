// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/type.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API MoveEltwiseUpThroughDataMov : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("MoveEltwiseUpThroughDataMov", "0");
    MoveEltwiseUpThroughDataMov(std::vector<DiscreteTypeInfo> allowed_data_movement_ops = get_default_allowed_ops());

private:
    static std::vector<DiscreteTypeInfo> get_default_allowed_ops();
};

}  // namespace pass
}  // namespace ov
