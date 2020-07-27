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

    class TRANSFORMATIONS_API ConvertTensorIteratorToSequence;

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
 * TODO:fill
 *
 */

class ngraph::pass::ConvertTensorIteratorToSequence: public ngraph::pass::GraphRewrite, public ngraph::pass::PassParam {
public:
    ConvertTensorIteratorToSequence() : GraphRewrite(), PassParam() {
        convert_ti_to_sequence();
    }

private:
    void convert_ti_to_sequence();
};
