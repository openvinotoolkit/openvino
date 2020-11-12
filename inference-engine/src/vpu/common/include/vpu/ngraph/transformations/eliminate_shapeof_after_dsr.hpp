// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace vpu {

class EliminateShapeOfAfterDSR : public ngraph::pass::GraphRewrite {
public:
    NGRAPH_RTTI_DECLARATION;
    EliminateShapeOfAfterDSR();
};

} //namespace vpu
