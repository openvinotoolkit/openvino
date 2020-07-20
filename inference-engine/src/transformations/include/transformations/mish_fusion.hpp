// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API MishFusion;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::MishFusion: public ngraph::pass::GraphRewrite {
public:
    MishFusion() : GraphRewrite() {
        mish_fusion()();
    }

private:
    void mish_fusion();
};
