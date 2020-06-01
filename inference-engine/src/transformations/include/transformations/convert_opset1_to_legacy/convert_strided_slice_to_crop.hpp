// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>
#include <algorithm>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API ConvertStridedSliceToCrop;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertStridedSliceToCrop: public ngraph::pass::GraphRewrite {
public:
    ConvertStridedSliceToCrop() : GraphRewrite() {
        convert_strided_slice_to_crop();
    }

private:
    void convert_strided_slice_to_crop();
};
