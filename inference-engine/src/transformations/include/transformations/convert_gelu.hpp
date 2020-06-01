// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

#include "ngraph/op/fused/gelu.hpp"
#include "transformations/utils/pass_param.hpp"

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API ConvertGELU;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertGELU: public ngraph::pass::GraphRewrite, public ngraph::pass::PassParam {
public:
    ConvertGELU() : GraphRewrite(), PassParam() {
        convert_gelu();
    }

private:
    void convert_gelu();
};
