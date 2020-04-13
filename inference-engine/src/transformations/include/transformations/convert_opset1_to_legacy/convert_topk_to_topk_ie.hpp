// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>
#include <string>

#include <ie_api.h>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class INFERENCE_ENGINE_API_CLASS(ConvertTopKToTopKIE);

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertTopKToTopKIE : public ngraph::pass::GraphRewrite {
public:
    ConvertTopKToTopKIE() : GraphRewrite() {
        convert_topk_to_topk_ie();
    }

private:
    void convert_topk_to_topk_ie();
};
