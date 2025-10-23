// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations/snippets/x64/op/brgemm_utils.hpp"

namespace ov::intel_cpu::pass {

/**
 * @interface FuseBrgemmCPUPostops
 * @brief FuseBrgemmCPUPostops applies a series of transformations to fuse supported post-operations into BrgemmCPU
 * nodes. This includes a bunch of eltwises and convert operations that can be executed as part of the Brgemm kernel.
 *
 * @ingroup snippets
 */
class FuseBrgemmCPUPostops : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("FuseBrgemmCPUPostops");
    explicit FuseBrgemmCPUPostops(std::set<size_t>& brgemm_external_params_idces);
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;

    static bool can_be_fused_as_postop(const std::shared_ptr<const ov::Node>& node);
    static bool brgemm_can_fuse_postop(const ov::element::Type& input_precision);

private:
    std::set<size_t>& m_brgemm_external_params_idces;
    // Note: this set is needed to collect external params.
    // This set will be converted to m_external_params_indices at run_on_model stage
    std::set<std::shared_ptr<ov::op::v0::Parameter>> m_external_params;
};

/**
 * @interface FuseConvert
 * @brief FuseConvert identifies ConvertSaturation nodes following BrgemmCPU and fuses them into the BrgemmCPU node.
 *
 * @ingroup snippets
 */
class FuseConvert : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FuseConvert");
    FuseConvert();
};

/**
 * @interface FuseUnaryEltwise
 * @brief FuseUnaryEltwise fuses unary eltwise operations into BrgemmCPU nodes.
 *
 * @ingroup snippets
 */
class FuseUnaryEltwise : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FuseUnaryEltwise");
    FuseUnaryEltwise();

    static bool can_be_fused(const std::shared_ptr<const ov::Node>& node);
};

/**
 * @interface FuseScalarEltwise
 * @brief FuseScalarEltwise fuses eltwise operations with scalar inputs into BrgemmCPU nodes.
 *
 * @ingroup snippets
 */
class FuseScalarEltwise : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FuseScalarEltwise");
    FuseScalarEltwise();

    static bool can_be_fused(const std::shared_ptr<const ov::Node>& node);
};

/**
 * @interface FuseBinaryEltwise
 * @brief FuseBinaryEltwise fuses eltwise operations with non-scalar inputs into BrgemmCPU nodes.
 * It also handles external parameters for binary post-operations.
 *
 * @ingroup snippets
 */
class FuseBinaryEltwise : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FuseBinaryEltwise");
    explicit FuseBinaryEltwise(std::set<std::shared_ptr<ov::op::v0::Parameter>>& external_params);

    static bool can_be_fused(const std::shared_ptr<const ov::Node>& node);

private:
    size_t m_fused_postops_count = 0;
    std::set<std::shared_ptr<ov::op::v0::Parameter>>& m_external_params;

    static ov::pass::pattern::op::Predicate binary_input_predicate;
};

/**
 * @interface FuseScaleShift
 * @brief FuseScaleShift identifies sequences of Multiply and Add operations
 * and fuses them into BrgemmCPU nodes as a single eltwise_linear.
 *
 * @ingroup snippets
 */
class FuseScaleShift : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FuseScaleShift");
    FuseScaleShift();
};

/**
 * @interface FuseClip
 * @brief FuseClip identifies sequences of Maximum and Minimum operations
 * and fuses them into BrgemmCPU nodes as a single eltwise_clip
 *
 * @ingroup snippets
 */
class FuseClip : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FuseClip");
    FuseClip();
};

}  // namespace ov::intel_cpu::pass
