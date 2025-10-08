// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "memory_access.hpp"
#include "openvino/op/op.hpp"
#include "snippets/shape_inference/shape_inference.hpp"
#include "snippets/snippets_visibility.hpp"

namespace ov::snippets::op {

/**
 * @interface Brgemm
 * @brief Brgemm is a batch-reduced matrix multiplication with the support of arbitrary strides between matrices rows
 * @ingroup snippets
 */
class SNIPPETS_API Brgemm : virtual public modifier::MemoryAccess, public ov::op::Op {
public:
    OPENVINO_OP("Brgemm", "SnippetsOpset");
    explicit Brgemm(const Output<Node>& A,
                    const Output<Node>& B,
                    size_t offset_a = 0LU,
                    size_t offset_b = 0LU,
                    size_t offset_c = 0LU,
                    const std::vector<size_t>& layout_a = {},
                    const std::vector<size_t>& layout_b = {},
                    const std::vector<size_t>& layout_c = {});
    explicit Brgemm(const Output<Node>& A,
                    const Output<Node>& B,
                    const PortDescriptor& desc_a,
                    const PortDescriptor& desc_b,
                    const PortDescriptor& desc_c,
                    const std::vector<size_t>& layout_a = {},
                    const std::vector<size_t>& layout_b = {},
                    const std::vector<size_t>& layout_c = {});
    Brgemm() = default;

    size_t get_offset_a() const {
        return get_input_offset(0);
    }
    size_t get_offset_b() const {
        return get_input_offset(1);
    }
    size_t get_offset_c() const {
        return get_output_offset(0);
    }

    static ov::element::Type get_output_type(const ov::element::Type& in_type0, const ov::element::Type& in_type1);

    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool has_evaluate() const override {
        return false;
    }
    bool visit_attributes(AttributeVisitor& visitor) override;

protected:
    virtual ov::element::Type get_output_type() const;
    static std::vector<ov::PartialShape> get_planar_input_shapes(const std::vector<ov::Input<ov::Node>>& inputs);
    static ov::PartialShape infer_output_partial_shape(const std::vector<ov::PartialShape>& input_shapes);
    ov::PartialShape get_planar_output_shape(const ov::PartialShape& output_shape) const;

private:
    void custom_constructor_validate_and_infer_types(const std::vector<size_t>& layout_a,
                                                     const std::vector<size_t>& layout_b,
                                                     const std::vector<size_t>& layout_c);
};

}  // namespace ov::snippets::op
