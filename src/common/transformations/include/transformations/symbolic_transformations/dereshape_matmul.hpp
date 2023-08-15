// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <openvino/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>

namespace ov {
namespace pass {
class TRANSFORMATIONS_API DeReshapeMatMul;
}  // namespace pass
}  // namespace ov

class ov::pass::DeReshapeMatMul : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("DeReshapeMatMul", "0");
    DeReshapeMatMul();
};
