// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace GNAPluginNS {

/**
 * @brief Convert a depthwise separable convolution (represented by a GroupConvolution) to a set of ScaleShift layers (MatMul + Add)
 */
class ConvertDWSCToScaleShifts : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertDWSCToScaleShifts();
};

/**
 * @brief Convert a depthwise separable convolution with bias (represented by a GroupConvolution + Add) to a set of ScaleShift layers (MatMul + Add)
 */
class ConvertDWSCBiasToScaleShifts : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertDWSCBiasToScaleShifts();
};

/**
 * @brief Convert a depthwise separable convolution + potential bias (represented by a GroupConvolution + Add), processed by POT,
 * to a set of ScaleShift layers (MatMul + Add)
 */
class ConvertDWSCWithFqToScaleShifts : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertDWSCWithFqToScaleShifts();
};

} // namespace GNAPluginNS
