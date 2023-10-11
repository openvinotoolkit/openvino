// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace intel_gna {
namespace pass {

/**
 * @brief Substitites ngraph::Convolution (NCHW) -> GNAConvolution (NHWC)
 *
 *                              Transpose (NCHW -> NHWC)
 *                                       |
 * Convolution (NCHW) ->         GNAConvolution (NHWC)
 *                                       |
 *                              Transpose (NHWC -> NCHW)
 */
class SubstituteGNAConvolution : public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    SubstituteGNAConvolution();
};

/**
 * @brief Substitites ngraph::MaxPool (NCHW) -> GNAMaxPool (NHWC)
 *
 *                              Transpose (NCHW -> NHWC)
 *                                       |
 * MaxPool (NCHW) ->               GNAMaxPool (NHWC)
 *                                       |
 *                              Transpose (NHWC -> NCHW)
 */
class SubstituteGNAMaxPool : public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    SubstituteGNAMaxPool();
};

/**
 * @brief calls SubstituteGNAConvolution and SubstituteGNAMaxPool together
 */
class ReplaceGnaNHWCLayers : public ov::pass::ModelPass {
public:
    NGRAPH_RTTI_DECLARATION;
    bool run_on_model(const std::shared_ptr<ngraph::Function>& f) override;
};

}  // namespace pass
}  // namespace intel_gna
}  // namespace ov
