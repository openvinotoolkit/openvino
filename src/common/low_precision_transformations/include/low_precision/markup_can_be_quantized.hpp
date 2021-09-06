// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/pass/pass.hpp>
#include "low_precision/lpt_visibility.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

class LP_TRANSFORMATIONS_API MarkupCanBeQuantized;

}  // namespace low_precision
}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief MarkupCanBeQuantized transformation marks Convolution, ConvolutionBackpropData, GroupConvolution and Concat
 * operations as can be quantized or not. If an operation is not quantized, then PrecisionsAttribute attribute instance
 * is created with empty precisions.
 */
class ngraph::pass::low_precision::MarkupCanBeQuantized : public ngraph::pass::FunctionPass {
public:
    NGRAPH_RTTI_DECLARATION;
    bool run_on_model(const std::shared_ptr<ngraph::Function>& m) override;
};
