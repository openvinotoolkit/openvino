// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>

namespace ov {
namespace intel_gpu {

class ConvertAvgPoolingToReduce : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertAvgPoolingToReduce", "0");
    ConvertAvgPoolingToReduce();
};

}  // namespace pass
}  // namespace ngraph
