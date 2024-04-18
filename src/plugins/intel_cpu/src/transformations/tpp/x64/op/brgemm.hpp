// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "transformations/snippets/x64/op/brgemm_cpu.hpp"
#include "modifiers.hpp"

namespace ov {
namespace intel_cpu {
namespace tpp {
namespace op {

/**
 * @interface BrgemmTPP
 * @brief BrgemmTPP is a batch-reduced matrix multiplication with the support of arbitrary strides between matrices rows
 *        with support of several precisions on plugin level
 * @ingroup snippets
 */
class BrgemmTPP : virtual public modifier::TensorProcessingPrimitive, public snippets::op::Brgemm {
public:
    OPENVINO_OP("Brgemm", "TppOpset", snippets::op::Brgemm);

    BrgemmTPP(const Output<Node>& A, const Output<Node>& B,
              size_t offset_a = 0, size_t offset_b = 0, size_t offset_c = 0,
              std::vector<size_t> layout_a = {}, std::vector<size_t> layout_b = {}, std::vector<size_t> layout_c = {},
              size_t blk_size_m = 0, size_t blk_size_k = 0, size_t blk_size_n = 0, float beta = 1);
    BrgemmTPP(const Output<Node>& A, const Output<Node>& B,
              const PortDescriptor& desc_a, const PortDescriptor& desc_b, const PortDescriptor& desc_c,
              std::vector<size_t> layout_a = {}, std::vector<size_t> layout_b = {}, std::vector<size_t> layout_c = {},
              size_t blk_size_m = 0, size_t blk_size_k = 0, size_t blk_size_n = 0, float beta = 1);
    BrgemmTPP() = default;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool visit_attributes(AttributeVisitor& visitor) override;
};

} // namespace op
} // namespace tpp
} // namespace intel_cpu
} // namespace ov
