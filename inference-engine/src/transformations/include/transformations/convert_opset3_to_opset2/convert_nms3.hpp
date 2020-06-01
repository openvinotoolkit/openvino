// Copyright (C) 2018-2020 Intel Corporation
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

class TRANSFORMATIONS_API ConvertNMS3;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertNMS3: public ngraph::pass::GraphRewrite, public ngraph::pass::PassParam {
public:
    ConvertNMS3() : GraphRewrite() {
        convert_nms3();
    }

private:
    void convert_nms3();
};
