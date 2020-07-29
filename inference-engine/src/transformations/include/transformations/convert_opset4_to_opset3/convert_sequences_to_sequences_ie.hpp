// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>
#include <string>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API ConvertLSTMSequenceMatcher;
class TRANSFORMATIONS_API ConvertGRUSequenceMatcher;
class TRANSFORMATIONS_API ConvertRNNSequenceMatcher;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertLSTMSequenceMatcher : public ngraph::pass::MatcherPass {
public:
    ConvertLSTMSequenceMatcher();
};

class ngraph::pass::ConvertGRUSequenceMatcher : public ngraph::pass::MatcherPass {
public:
    ConvertGRUSequenceMatcher();
};

class ngraph::pass::ConvertRNNSequenceMatcher : public ngraph::pass::MatcherPass {
public:
    ConvertRNNSequenceMatcher();
};
