// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/op/memory_access.hpp"
#include "openvino/op/op.hpp"

#include "brgemm_utils.hpp"

namespace ov {
namespace intel_cpu {

/**
* @interface BrgemmCopyA
* @brief The operation for data repacking of first input of Brgemm
* @ingroup snippets
*/
class BrgemmCopyA : public snippets::modifier::MemoryAccess, public ov::op::Op {
public:
    using BrgemmConfig = brgemm_utils::BrgemmConfig;
    OPENVINO_OP("BrgemmCopyA", "SnippetsOpset");

    BrgemmCopyA(const Output<Node>& x, BrgemmConfig config, const size_t offset_in = 0lu, const size_t offset_out = 0lu, std::vector<size_t> layout_in = {});
    BrgemmCopyA(const Output<Node>& x, BrgemmConfig config, const PortDescriptor& desc_in, const PortDescriptor& desc_out, std::vector<size_t> layout_in = {});
    BrgemmCopyA() = default;

    size_t get_offset_in() const { return get_input_offset(0); }
    size_t get_offset_out() const { return get_output_offset(0); }

    const BrgemmConfig& get_config() const { return m_config; }

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    bool has_evaluate() const override { return false; }
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    class ShapeInfer : public snippets::IShapeInferSnippets {
        std::vector<size_t> m_layout {};
    public:
        explicit ShapeInfer(const std::shared_ptr<ov::Node>& n);
        Result infer(const std::vector<snippets::VectorDimsRef>& input_shapes) override;
    };

private:
    void custom_constructor_validate_and_infer_types(std::vector<size_t> layout_input = {});

    const BrgemmConfig m_config {};
};
} // namespace intel_cpu
} // namespace ov
