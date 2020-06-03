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

class TRANSFORMATIONS_API ConvertProposalToProposalIE;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertProposalToProposalIE: public ngraph::pass::GraphRewrite {
public:
    ConvertProposalToProposalIE() : GraphRewrite() {
        convert_proposal();
    }

private:
    void convert_proposal();
};
