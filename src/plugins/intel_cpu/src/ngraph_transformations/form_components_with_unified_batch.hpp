// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/pass/graph_rewrite.hpp>

namespace ov {
namespace intel_cpu {

class FormComponentsWithUnifiedBatch : public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    FormComponentsWithUnifiedBatch();
};

}  // namespace intel_cpu
}  // namespace ov