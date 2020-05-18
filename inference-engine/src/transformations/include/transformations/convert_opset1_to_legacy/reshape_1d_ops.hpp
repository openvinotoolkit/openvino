// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <ie_api.h>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class INFERENCE_ENGINE_API_CLASS(Reshape1DOps);

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