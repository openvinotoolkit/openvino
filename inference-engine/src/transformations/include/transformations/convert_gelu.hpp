// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <ie_api.h>

#include <ngraph/pass/graph_rewrite.hpp>

#include "ngraph/op/fused/gelu.hpp"
#include "transformations/utils/pass_param.hpp"

namespace ngraph {
namespace pass {

class INFERENCE_ENGINE_API_CLASS(ConvertGELU);

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
