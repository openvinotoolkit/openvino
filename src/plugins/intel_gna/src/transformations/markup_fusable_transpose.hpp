// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace intel_gna {
namespace pass {

/**
 * @brief Markup fusable tranpose
 * This transformation is written to support old IRs for Kaldi models
 * with specific 0-3-2-1 transpose after Convolution and mark it up
 * for special handling in compiler for backward compatibility purposes
 */
class MarkupFusableTranspose : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("MarkupFusableTranspose", "0");
    bool run_on_model(const std::shared_ptr<ngraph::Function>& f) override;
};

}  // namespace pass
}  // namespace intel_gna
}  // namespace ov