// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include <ie_api.h>

#include <ngraph/ngraph.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

using namespace std;

namespace ngraph {
namespace pass {

class INFERENCE_ENGINE_API_CLASS(BatchNormDecomposition);

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::BatchNormDecomposition: public ngraph::pass::GraphRewrite {
public:
    BatchNormDecomposition() : GraphRewrite() {
        batch_norm_decomposition();
    }

private:
    void batch_norm_decomposition();
};
