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

class INFERENCE_ENGINE_API_CLASS(ConvertPReLUToReLUIE);

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertPReLUToReLUIE: public ngraph::pass::GraphRewrite {
public:
    ConvertPReLUToReLUIE() : GraphRewrite() {
        convert_prelu();
    }

private:
    void convert_prelu();
};
