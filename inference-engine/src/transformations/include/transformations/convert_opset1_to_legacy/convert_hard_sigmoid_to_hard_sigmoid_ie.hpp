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

class TRANSFORMATIONS_API ConvertHardSigmoidToHardSigmoidIE;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertHardSigmoidToHardSigmoidIE : public ngraph::pass::GraphRewrite {
public:
    ConvertHardSigmoidToHardSigmoidIE() : GraphRewrite() {
        convert_hard_sigmoid();
    }

private:
    void convert_hard_sigmoid();
};
