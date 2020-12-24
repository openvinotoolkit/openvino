// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "legacy/transformations/convert_opset1_to_legacy/convert_mul_or_add_finally.hpp"

NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvertMulOrAddFinally, "ConvertMulOrAddFinally", 0);

ngraph::pass::ConvertMulOrAddFinally::ConvertMulOrAddFinally() : GraphRewrite() {
    convert_mul_or_add_finally<ngraph::opset1::Add>();
    convert_mul_or_add_finally<ngraph::opset1::Subtract>();
    convert_mul_or_add_finally<ngraph::opset1::Multiply>();
}
