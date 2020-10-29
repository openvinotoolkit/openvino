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

class INFERENCE_ENGINE_API_CLASS(ConvertStridedSliceToStridedSliceIE);

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertStridedSliceToStridedSliceIE: public ngraph::pass::GraphRewrite {
public:
    ConvertStridedSliceToStridedSliceIE() : GraphRewrite() {
        convert_strided_slice_to_strided_slice_ie();
    }

private:
    void convert_strided_slice_to_strided_slice_ie();
};
