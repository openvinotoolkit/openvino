// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class MatMulBiasFusion;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::MatMulBiasFusion : public ngraph::pass::GraphRewrite {
public:
    MatMulBiasFusion() : GraphRewrite() {
        construct_matmulbias();
    }

private:
    void construct_matmulbias();
};
