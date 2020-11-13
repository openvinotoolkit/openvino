// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>
#include <algorithm>

#include <transformations_visibility.hpp>

#include <ngraph/pass/pass.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/validation_util.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API ConvertNMSIE3Precision;

}  // namespace pass
}  // namespace ngraph

/*
 * Description:
 *      Convert precision for NMSIE3.
 */


class ngraph::pass::ConvertNMSIE3Precision : public ngraph::pass::FunctionPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertNMSIE3Precision()
        : FunctionPass()
        {}

    bool run_on_function(std::shared_ptr<Function> f) override;
};
