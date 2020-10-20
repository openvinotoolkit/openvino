// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/batch_norm_decomposition.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <ngraph/rt_info.hpp>

using namespace ngraph;

NGRAPH_RTTI_DEFINITION(ngraph::pass::BatchNormDecomposition, "BatchNormDecomposition", 0);

ngraph::pass::BatchNormDecomposition::BatchNormDecomposition() {
    register_batch_norm_matcher<opset1::BatchNormInference>();
    register_batch_norm_matcher<opset5::BatchNormInference>();
}
