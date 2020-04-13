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

class INFERENCE_ENGINE_API_CLASS(Reshape1DConvolutions);

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::Reshape1DConvolutions: public ngraph::pass::GraphRewrite {
public:
    Reshape1DConvolutions() : GraphRewrite() {
        reshape_convolutions();
    }

private:
    void reshape_convolutions();
};
