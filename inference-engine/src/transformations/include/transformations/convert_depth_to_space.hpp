// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <ie_api.h>

#include <ngraph/pass/graph_rewrite.hpp>
#include "transformations/utils/pass_param.hpp"

namespace ngraph {
namespace pass {

class INFERENCE_ENGINE_API_CLASS(ConvertDepthToSpace);

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertDepthToSpace: public ngraph::pass::GraphRewrite, public ngraph::pass::PassParam {
public:
    ConvertDepthToSpace() : GraphRewrite(), PassParam() {
        convert_depth_to_space();
    }

private:
    void convert_depth_to_space();
};
