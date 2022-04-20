// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace vpu {

class EliminateShapeOfAfterDSR : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("EliminateShapeOfAfterDSR", "0");
    EliminateShapeOfAfterDSR();
};

} //namespace vpu
