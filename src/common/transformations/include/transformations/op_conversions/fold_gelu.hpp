// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API FoldGelu;

}  // namespace pass
}  // namespace ov

class ov::pass::FoldGelu : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("FoldGelu", "0");
    FoldGelu();
};
