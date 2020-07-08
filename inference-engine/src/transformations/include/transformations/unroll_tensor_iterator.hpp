// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>
#include "transformations/utils/pass_param.hpp"

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API UnrollTensorIterator;

}  // namespace pass
}  // namespace ngraph

/*
 * Description:
 * TODO:fill
 *
 * Usage:
 * TODO:fill
 *
 * Callback example:
 * TODO: fill
 *
 */

class ngraph::pass::UnrollTensorIterator: public ngraph::pass::GraphRewrite, public ngraph::pass::PassParam {
public:
    UnrollTensorIterator() : GraphRewrite(), PassParam() {
        unroll_tensor_iterator();
    }

private:
    void unroll_tensor_iterator();
};
