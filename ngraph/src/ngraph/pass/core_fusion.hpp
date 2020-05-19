//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include "ngraph/pass/graph_rewrite.hpp"

namespace ngraph
{
    namespace pass
    {
        class CoreFusion;
    }
}

class NGRAPH_API ngraph::pass::CoreFusion : public ngraph::pass::GraphRewrite
{
public:
    CoreFusion(FusionTypeMask fusions = FusionType::REGULAR_FUSIONS)
        : GraphRewrite()
    {
        if (fusions.is_set(FusionType::REGULAR_FUSIONS))
        {
            construct_relu();
            construct_folded_batch_norm();
            construct_conv_affine_folding();
            construct_sigmoid();
            construct_sigmoid_bprop();
            construct_optimized_strided_conv();
            construct_reshape_broadcast();
            construct_reshape_softmax_reshape();
            construct_zero_padded_reshaped_conv();
            construct_zero_padded_conv();
            construct_zero_padded_conv_backprop_filters();
            construct_softmax_cross_entropy_fprop();
            construct_softmax_cross_entropy_bprop_with_soft_labels();
            construct_softmax_cross_entropy_bprop_with_ignore_mask();
        }
        // Patterns under FOP_FUSIONS create ops (FusedOps) that might not
        // be all supported by certain backends. In such a case, backends
        // can register a FusedOpDecomposition pass after CoreFusion that will
        // selectively decompose the unsupported ops back to the Core opset
        if (fusions.is_set(FusionType::FOP_FUSIONS))
        {
            construct_conv_bias();
            construct_conv_bias_add();
        }
    }
    void construct_relu();
    void construct_folded_batch_norm();
    void construct_conv_affine_folding();
    void construct_sigmoid();
    void construct_sigmoid_bprop();
    void construct_optimized_strided_conv();
    void construct_reshape_broadcast();
    void construct_reshape_softmax_reshape();
    void construct_zero_padded_reshaped_conv();
    void construct_zero_padded_conv();
    void construct_zero_padded_conv_backprop_filters();
    void construct_conv_bias();
    void construct_conv_bias_add();
    void construct_softmax_cross_entropy_fprop();
    void construct_softmax_cross_entropy_bprop_with_soft_labels();
    void construct_softmax_cross_entropy_bprop_with_ignore_mask();
};
