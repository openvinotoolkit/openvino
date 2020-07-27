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

class TRANSFORMATIONS_API ConvertNMS1ToNMS3;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertNMS1ToNMS3: public ngraph::pass::GraphRewrite {
public:
    ConvertNMS1ToNMS3() : GraphRewrite() {
        convert_nms1_to_nms3();
    }

private:
    void convert_nms1_to_nms3();
};
