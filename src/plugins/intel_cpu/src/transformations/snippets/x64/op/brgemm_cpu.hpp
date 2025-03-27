// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <common/primitive_attr.hpp>

#include "brgemm_copy_b.hpp"
#include "brgemm_utils.hpp"
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
    using BRGEMM_TYPE = brgemm_utils::BRGEMM_TYPE;
    OPENVINO_OP("BrgemmCPU", "SnippetsOpset", snippets::op::Brgemm);

    struct PostopsConfig {
        dnnl_post_ops post_ops = {};
        int binary_postops_offset = -1;

        PostopsConfig() : post_ops({}), binary_postops_offset(-1) {}
    };
    BrgemmCPU(const ov::OutputVector& inputs,
              BRGEMM_TYPE type,
              const std::vector<PortDescriptor>& input_descs = {},
              const PortDescriptor& output_desc = {0, 0},
              const std::vector<size_t>& layout_a = {},
              const std::vector<size_t>& layout_b = {},
              const std::vector<size_t>& layout_c = {},
              const PostopsConfig& post_ops = PostopsConfig{});
    BrgemmCPU() = default;

    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    BRGEMM_TYPE get_type() const {
        return m_type;
    }

    size_t get_offset_scratch() const;

    const PostopsConfig& get_postops_config() const {
        return m_post_ops;
    }

    size_t get_main_inputs_count() const {
        return m_main_inputs_count;
    }

    ov::OutputVector get_postop_inputs() const;

    bool visit_attributes(AttributeVisitor& visitor) override;

    constexpr static size_t SCRATCH_BYTE_SIZE = 32 * 1024;

private:
    void custom_constructor_validate_and_infer_types(const std::vector<size_t>& layout_a,
                                                     const std::vector<size_t>& layout_b,
                                                     const std::vector<size_t>& layout_c);
    static size_t compute_main_inputs_count(const BRGEMM_TYPE type);
    void validate_with_scratchpad() const;
    void validate_inputs() const;
    ov::element::Type get_output_type() const override;

    BRGEMM_TYPE m_type = BRGEMM_TYPE::STAND_ALONE;

    PostopsConfig m_post_ops = {};

    ov::element::Type m_forced_output_type = ov::element::undefined;

    const size_t m_main_inputs_count = 0lu;
};
}  // namespace ov::intel_cpu
