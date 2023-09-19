// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "subgraph_pass.hpp"

namespace ov {
namespace snippets {
namespace pass {

/**
 * @interface SplitDimensionM
 * @brief Inserts Reshape nodes after and before Parameters and Results in Subgraphs with MatMul inside
 *        to split dimension M for MatMuls to increase work amount for parallelism
 *        Note: works only with 3D MHA patterns
 * @ingroup snippets
 */
class SplitDimensionM: public CommonOptimizations::SubgraphPass {
public:
    OPENVINO_RTTI("SplitDimensionM", "0");
    SplitDimensionM(size_t concurrency) : m_concurrency(concurrency) {}

    // Return True if the MatMul node is supported by this optimization
    static bool isSupportedMatMul(const std::shared_ptr<const ov::Node>& node);

    // Returns True if parallelism work amount (concurrency) can be increased by this optimization
    static bool canBeOptimized(const std::shared_ptr<const ov::Node>& node, size_t concurrency);

    bool run_on_subgraph(const std::shared_ptr<op::Subgraph>& subgraph) override;

private:
    size_t m_concurrency;
};


} // namespace pass
} // namespace snippets
} // namespace ov
