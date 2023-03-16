// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"
#include "memory_access.hpp"

namespace ngraph {
namespace snippets {
namespace op {

/**
 * @interface Brgemm
 * @brief Brgemm is a batch-reduced matrix multiplication with the support of arbitrary strides between matrices rows
 * @ingroup snippets
 */
class Brgemm : public MemoryAccess {
public:
    OPENVINO_OP("Brgemm", "SnippetsOpset", MemoryAccess);
    Brgemm(const Output<Node>& A, const Output<Node>& B,
           const size_t offset_a = 0lu, const size_t offset_b = 0lu, const size_t offset_c = 0lu);
    Brgemm() = default;

    size_t get_offset_a() const { return get_input_offset(0); }
    size_t get_offset_b() const { return get_input_offset(1); }
    size_t get_offset_c() const { return get_output_offset(0); }

    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool has_evaluate() const override { return false; }

protected:
    ov::element::Type get_output_type() const;
    ov::PartialShape get_output_partial_shape(const std::vector<ov::PartialShape>& input_shapes) const;
};

} // namespace op
} // namespace snippets
} // namespace ngraph