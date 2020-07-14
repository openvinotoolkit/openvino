// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>
#include <algorithm>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/validation_util.hpp>
#include <ngraph/rt_info.hpp>


namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API ConvertPowerSumAddRsqrtToNormalizeL2;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertPowerSumAddRsqrtToNormalizeL2: public ngraph::pass::GraphRewrite {
public:
    ConvertPowerSumAddRsqrtToNormalizeL2() : GraphRewrite() {
        convert_to_normalize_l2();
    }

private:
    void convert_to_normalize_l2();
    bool is_applicable(pattern::Matcher& m);
};
