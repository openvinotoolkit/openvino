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

class TRANSFORMATIONS_API Reshape1DOps;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::Reshape1DOps: public ngraph::pass::GraphRewrite {
public:
    Reshape1DOps() : GraphRewrite() {
        reshape_ops();
    }

private:
    void reshape_ops();
};
