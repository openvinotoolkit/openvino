// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

    class TRANSFORMATIONS_API ConvertTensorIteratorToLSTMSequence;
    class TRANSFORMATIONS_API ConvertTensorIteratorToRNNSequence;
    class TRANSFORMATIONS_API ConvertTensorIteratorToGRUSequence;

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

class ngraph::pass::ConvertTensorIteratorToLSTMSequence: public ngraph::pass::MatcherPass {
public:
    ConvertTensorIteratorToLSTMSequence();
};

class ngraph::pass::ConvertTensorIteratorToRNNSequence: public ngraph::pass::MatcherPass {
public:
    ConvertTensorIteratorToRNNSequence();
};

class ngraph::pass::ConvertTensorIteratorToGRUSequence: public ngraph::pass::MatcherPass {
public:
    ConvertTensorIteratorToGRUSequence();
};
