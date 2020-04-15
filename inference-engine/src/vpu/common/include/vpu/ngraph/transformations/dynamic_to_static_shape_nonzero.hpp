// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

#include <vector>
#include <memory>

namespace ngraph {
namespace pass {

class DynamicToStaticShapeNonZero : public GraphRewrite {
public:
    DynamicToStaticShapeNonZero();
};

}  // namespace pass
}  // namespace ngraph
