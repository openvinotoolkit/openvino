// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace GNAPluginNS {

/**
 * @brief Decompose MVN operation
 * See official OpenVINO documentation for the MVN formula
 * implemented partially by this decomposition:
 * https://docs.openvino.ai/latest/openvino_docs_ops_normalization_MVN_6.html
 *
 */
class DecomposeMVN : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("DecomposeMVN", "0");
    DecomposeMVN();
};

}  // namespace GNAPluginNS
