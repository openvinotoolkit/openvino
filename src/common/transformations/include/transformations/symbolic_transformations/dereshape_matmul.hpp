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
class TRANSFORMATIONS_API DeReshapeMatMulWithComplications;
}  // namespace pass
}  // namespace ov

class ov::pass::DeReshapeMatMul : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("DeReshapeMatMul", "0");
    DeReshapeMatMul();
};

class ov::pass::DeReshapeMatMulWithComplications : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("DeReshapeMatMulWithComplications", "0");
    DeReshapeMatMulWithComplications();
};
