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

/*
 * Description:
 *     This transformation converts opset1::StridedSlice to legacy StridedSliceIE
 *     StridedSliceIE takes begin, end and strides inputs ony in i32 precision.
 *     Inputs with precision != i32 are converted with Convert operation.
 */

class ngraph::pass::ConvertStridedSliceToStridedSliceIE: public ngraph::pass::GraphRewrite {
public:
    ConvertStridedSliceToStridedSliceIE() : GraphRewrite() {
        convert_strided_slice_to_strided_slice_ie();
    }

private:
    void convert_strided_slice_to_strided_slice_ie();
};
