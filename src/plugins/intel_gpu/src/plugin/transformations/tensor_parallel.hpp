// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
/**
 * @ingroup ov_transformation_common_api
 * @brief TensorParallelFusion transformation matches following graph:
 *
 *         +----------+            +----------+
 *         |    A     |            |  Weights |
 *         +----------+            +----------+
 *              |                       |
 *              -----------    ----------
 *                        |    |
 *                        v    v
 *                      +--------+
 *                      |   FC   |
 *                      +--------+
 *                          |
 *                          v
 *
 *
 * and replaces with:
 *
 *                           +-------+   +----------+
 *                           |   A   |   | Weights  |
 *                           +-------+   +----------+
 *                                |            |
 *                                ------  ------
 *                                     |  |
 *                                     v  v
 *         +----------+            +----------+
 *         |   sync   |            |    FC    |
 *         +----------+            +----------+
 *              |                       |
 *              -----------    ----------
 *                        |    |
 *                        v    v
 *                      +--------+
 *                      | Reduce |
 *                      +--------+
 */
// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

namespace ov {
namespace intel_gpu {

class TensorParallelFusion: public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertMatMulToFullyConnected", "0");
    TensorParallelFusion();
};

}   // namespace intel_gpu
}   // namespace ov
