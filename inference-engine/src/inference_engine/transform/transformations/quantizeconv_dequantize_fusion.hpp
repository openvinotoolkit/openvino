// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class QuantizeConvDequantizeFusion;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::QuantizeConvDequantizeFusion: public ngraph::pass::GraphRewrite {
public:
    QuantizeConvDequantizeFusion() : GraphRewrite() {
        quantize_conv_fusion();
    }

private:
    void quantize_conv_fusion();
};
