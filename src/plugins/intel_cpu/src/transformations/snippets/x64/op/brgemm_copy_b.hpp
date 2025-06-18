// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "brgemm_utils.hpp"
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/op.hpp"
#include "snippets/op/memory_access.hpp"
#include "snippets/shape_inference/shape_inference.hpp"
#include "snippets/shape_types.hpp"

namespace ov::intel_cpu {

/**
* @interface BrgemmCopyB
* @brief The operation for data repacking of Brgemm with input non-fp32 precisions.
         The CPU Generator uses oneDNN primitives for generation code of Brgemm.
         OneDNN requiers data repacking for second input of Brgemm with input non-fp32 precisions.
* @ingroup snippets
*/
class BrgemmCopyB : public snippets::modifier::MemoryAccess, public ov::op::Op {
public:
    using BrgemmConfig = brgemm_utils::BrgemmConfig;
    OPENVINO_OP("BrgemmCopyB", "SnippetsOpset");

    BrgemmCopyB(const Output<Node>& x,
                const BrgemmConfig& config,
                const std::vector<size_t>& layout_input = {},
                const MemoryAccess::PortDescriptor& desc_in = {},
                const MemoryAccess::PortDescriptor& desc_out0 = {},
                const MemoryAccess::PortDescriptor& desc_out1 = {});
    BrgemmCopyB();

    size_t get_offset_in() const {
        return get_input_offset(0);
    }
    size_t get_offset_out() const {
        return get_output_offset(0);
    }
    size_t get_offset_compensations() const;

    const BrgemmConfig& get_config() const {
        return m_config;
    }

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    class ShapeInfer : public snippets::IShapeInferSnippets {
        std::vector<size_t> m_layout;
        size_t m_num_outs = 1;

    public:
        explicit ShapeInfer(const std::shared_ptr<ov::Node>& n);
        Result infer(const std::vector<snippets::VectorDimsRef>& input_shapes) override;
    };

    static bool is_transposed(const std::vector<size_t>& layout);

private:
    void custom_constructor_validate_and_infer_types(const std::vector<size_t>& layout_input = {});

    const BrgemmConfig m_config;
};
}  // namespace ov::intel_cpu
