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

class TRANSFORMATIONS_API ConvertConvolutions;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertConvolutions: public ngraph::pass::GraphRewrite {
public:
    ConvertConvolutions() : GraphRewrite() {
        convert_convolution();
        convert_group_convolution();
        convert_convolution_backprop_data();
        convert_group_convolution_backprop_data();
    }

private:
    void convert_convolution();

    void convert_group_convolution();

    void convert_convolution_backprop_data();

    void convert_group_convolution_backprop_data();
};
