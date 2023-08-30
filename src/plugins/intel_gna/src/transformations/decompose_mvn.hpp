// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace intel_gna {
namespace pass {

/**
 * @brief Decompose MVN operation
 * See official OpenVINO documentation for the MVN formula
 * implemented partially by this decomposition:
 * https://docs.openvino.ai/2023.0/openvino_docs_ops_normalization_MVN_6.html
 *
 */
class DecomposeMVN : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("DecomposeMVN", "0");
    DecomposeMVN();
};

}  // namespace pass
}  // namespace intel_gna
}  // namespace ov
