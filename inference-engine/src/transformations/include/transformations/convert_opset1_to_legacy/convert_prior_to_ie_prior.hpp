// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <ie_api.h>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class INFERENCE_ENGINE_API_CLASS(ConvertPriorBox);

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertPriorBox: public ngraph::pass::GraphRewrite {
public:
    ConvertPriorBox() : GraphRewrite() {
        convert_prior_box();
        convert_prior_box_clustered();
    }

private:
    void convert_prior_box();

    void convert_prior_box_clustered();
};
