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
 * @brief Inserts Reshape nodes before inputs and after outputs of Subgraphs with MatMul inside
 *        to split dimension M for MatMuls. It allows to increase work amount for parallelism
 * @ingroup snippets
 */
class SplitDimensionM: public CommonOptimizations::SubgraphPass {
public:
    OPENVINO_RTTI("SplitDimensionM", "0");
    SplitDimensionM(size_t concurrency) : m_concurrency(concurrency) {}

    bool run_on_subgraph(const std::shared_ptr<op::Subgraph>& subgraph) override;

    // Return True if the MatMul node is supported by this optimization
    static bool is_supported_matmul(const std::shared_ptr<const ov::Node>& node);
    // Returns True if parallelism work amount (concurrency) can be increased by this optimization
    static bool can_be_optimized(const std::shared_ptr<const ov::Node>& node, size_t concurrency);

private:
    static std::shared_ptr<ov::op::v0::MatMul> get_matmul(const std::shared_ptr<op::Subgraph>& subgraph);
    static std::pair<size_t, size_t> get_splited_dimensions(size_t batch_dim, size_t m_dim, size_t optimal_parallelism_work_amount);
    static bool split(const ov::Shape& shape, size_t optimal_parallelism_work_amount, size_t& batch_m_dim, size_t& new_m_dim);

    void reshape_subgraph(const std::shared_ptr<op::Subgraph>& subgraph, const ov::Shape& shape, size_t batch_m_dim, size_t new_m_dim);

    size_t m_concurrency;
};


} // namespace pass
} // namespace snippets
} // namespace ov
