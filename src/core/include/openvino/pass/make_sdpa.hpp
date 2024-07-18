// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API MakeSDPA;

}  // namespace pass
}  // namespace ov

class ov::pass::MakeSDPA : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("MakeSDPA", "0");
    MakeSDPA();
};