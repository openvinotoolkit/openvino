// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace intel_gna {
namespace pass {

/**
 * @brief remove concat layers with single input
 *
 * Searches for next pattern
 *     Any input layer
 *           |
 *         Concat
 *           |
 *     Any output layer
 *
 * And transforms to
 *     Any input layer
 *           |
 *     Any output layer
 */
class RemoveSingleInputConcat : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("RemoveSingleInputConcat", "0");
    RemoveSingleInputConcat();
};

}  // namespace pass
}  // namespace intel_gna
}  // namespace ov
