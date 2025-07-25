// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/attribute_visitor.hpp"
#include "snippets/op/memory_access.hpp"
#include "snippets/shape_inference/shape_inference.hpp"
#include "snippets/shape_types.hpp"

namespace ov::intel_cpu::aarch64 {

/**
* @interface GemmCopyB
* @brief The operation for data repacking of GemmCPU.
         The CPU Generator uses kleidiAI primitives for generation code of Gemm.
         kleidiAI requiers data repacking for second input of Gemm.
* @ingroup snippets
*/
class GemmCopyB : public snippets::modifier::MemoryAccess, public ov::op::Op {
public:
    OPENVINO_OP("GemmCopyB", "SnippetsOpset");

    GemmCopyB(const Output<Node>& x,
              const PortDescriptor& desc_in0,
              const PortDescriptor& desc_out0,
              const std::vector<size_t>& layout_input = {});
    GemmCopyB() = default;

    size_t get_offset_in() const {
        return get_input_offset(0);
    }
    size_t get_offset_out() const {
        return get_output_offset(0);
    }

    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    class ShapeInfer : public snippets::IShapeInferSnippets {
        std::vector<size_t> m_layout;

    public:
        explicit ShapeInfer(const std::shared_ptr<ov::Node>& n);
        Result infer(const std::vector<snippets::VectorDimsRef>& input_shapes) override;
    };

private:
    void custom_constructor_validate_and_infer_types(const std::vector<size_t>& layout_input = {});
    static void validate_element_type(const ov::element::Type& element_type);
};
}  // namespace ov::intel_cpu::aarch64
