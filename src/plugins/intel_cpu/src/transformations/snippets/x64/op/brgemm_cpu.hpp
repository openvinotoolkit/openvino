// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <common/primitive_attr.hpp>
#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

#include "brgemm_copy_b.hpp"
#include "brgemm_utils.hpp"
#include "memory_desc/dnnl_memory_desc.h"
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/op/op.hpp"
#include "snippets/lowered/port_descriptor.hpp"
#include "snippets/op/brgemm.hpp"

namespace ov::intel_cpu {

/**
 * @interface BrgemmCPU
 * @brief BrgemmCPU is a batch-reduced matrix multiplication with the support of arbitrary strides between matrices rows
 *        with support of several precisions on plugin level
 * @ingroup snippets
 */
class BrgemmCPU : public snippets::op::Brgemm {
public:
    using BrgemmConfig = brgemm_utils::BrgemmConfig;
    OPENVINO_OP("BrgemmCPU", "SnippetsOpset", snippets::op::Brgemm);

    struct PostopsConfig {
        dnnl_post_ops post_ops;
        std::optional<size_t> binary_postops_offset = std::nullopt;
        std::optional<ov::element::Type> forced_output_type = std::nullopt;

        PostopsConfig();
        bool visit_attributes(AttributeVisitor& visitor);
    };
    BrgemmCPU(const ov::OutputVector& inputs,
              BrgemmConfig config,
              const std::vector<MemoryAccess::PortDescriptor>& input_descs = {},
              const MemoryAccess::PortDescriptor& output_desc = {0, 0},
              const std::vector<size_t>& layout_a = {},
              const std::vector<size_t>& layout_b = {},
              const std::vector<size_t>& layout_c = {},
              PostopsConfig post_ops = PostopsConfig{});
    BrgemmCPU();

    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    const BrgemmConfig& get_config() const {
        return m_config;
    }

    size_t get_offset_scratch() const;

    const PostopsConfig& get_postops_config() const {
        return m_post_ops_config;
    }

    size_t get_gemm_inputs_count() const {
        return m_gemm_inputs_count;
    }

    ov::OutputVector get_postop_inputs() const;

    /**
     * @brief Forces the output type of the BrgemmCPU operation via postops.
     */
    void force_output_type(const ov::element::Type& type);

    /**
     * @brief Adds a scalar unary, binary, and ternary post-operation (such as relu, round, linear, clip, etc.) to the
     * BrgemmCPU.
     * @param alg_kind The DNNL algorithm kind for the eltwise operation.
     * @param alpha The alpha parameter for the eltwise operation.
     * @param beta The beta parameter for the eltwise operation.
     */
    void add_scalar_eltwise_postop(dnnl::impl::alg_kind_t alg_kind, float alpha, float beta);

    /**
     * @brief Adds a binary eltwise post-operation (such as add, mul, max, min, etc.) to the BrgemmCPU.
     * @param alg_kind The DNNL algorithm kind for the binary operation.
     * @param desc The memory descriptor for the binary input.
     * @param postop_input The input node to be used in the binary operation.
     * @param binary_postop_offset The offset from base ptr of external ptr indices.
     */
    void add_binary_eltwise_postop(dnnl::impl::alg_kind_t alg_kind,
                                   const dnnl::memory::desc& desc,
                                   const ov::Output<Node>& postop_input,
                                   size_t binary_postop_offset);

    bool visit_attributes(AttributeVisitor& visitor) override;

    constexpr static size_t SCRATCH_BYTE_SIZE = 32 * 1024;

private:
    void custom_constructor_validate_and_infer_types(const std::vector<size_t>& layout_a,
                                                     const std::vector<size_t>& layout_b,
                                                     const std::vector<size_t>& layout_c);
    void validate_with_scratchpad() const;
    void validate_inputs_size() const;
    void validate_postop_inputs() const;
    ov::element::Type get_output_type() const override;

    /**
     * @brief Adds a new post op input to the BrgemmCPU operation. It also does the following:
     *       - Adds the new input to input Memory Access PortMap
     *       - Adds the new input to input port descriptors if they are already initialized
     *       - Validates the new input
     * @param postop_input The new input node to be added as a post-operation.
     */
    void add_postop_input(const ov::Output<Node>& postop_input);

    const BrgemmConfig m_config;

    PostopsConfig m_post_ops_config;

    /**
     * @brief m_gemm_inputs_count represents the number of GeMM inputs of the BrgemmCPU,
     *        which depends on the BRGEMM_TYPE. This count includes only the
     *        inputs needed directly for matrix multiplication execution.
     *        The rest inputs represents binary postops
     */
    const size_t m_gemm_inputs_count = 0LU;
};
}  // namespace ov::intel_cpu
