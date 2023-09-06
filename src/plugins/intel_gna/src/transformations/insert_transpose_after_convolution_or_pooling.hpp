// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace intel_gna {
namespace pass {

/**
 * @brief Inserts transpose after convolution or pooling if its output is reshaped to 3D or 4D data with only one last
 * dimension > 1 and the reshaped data is used as an input to the next convolution: Convolution / Pooling [1, C, 1, W]
 *                |
 *         Reshape [1, 1, 1, C*W]
 *                |
 *               ...
 *                |
 *           Convolution
 */
class InsertTransposeAfterConvOrPool : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("InsertTransposeAfterConvOrPool", "0");
    bool run_on_model(const std::shared_ptr<ngraph::Function>& f) override;
};

}  // namespace pass
}  // namespace intel_gna
}  // namespace ov
