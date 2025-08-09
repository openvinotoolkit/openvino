// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/op/op.hpp"
#include "snippets/op/memory_access.hpp"

namespace ov::snippets::op {

/**
 * @interface FA
 * @brief The operation represet a flash attention alg op
 * @ingroup snippets
 */
class FA : virtual public snippets::modifier::MemoryAccess, public ov::op::Op {
public:
    OPENVINO_OP("FA", "SnippetsOpset");

    FA(const Output<Node>& q, const Output<Node>& k, const Output<Node>& v);
    FA() = default;

    size_t get_offset_a() const {
        return get_input_offset(0);
    }
    size_t get_offset_b() const {
        return get_input_offset(1);
    }
    size_t get_offset_c() const {
        return get_input_offset(2);
    }
    size_t get_offset_d() const {
        return get_output_offset(0);
    }

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    void validate_and_infer_types() override;

protected:
    static ov::PartialShape infer_output_partial_shape(const std::vector<ov::PartialShape>& input_shapes);
    static std::vector<ov::PartialShape> get_planar_input_shapes(const std::vector<ov::Input<ov::Node>>& inputs);
    ov::PartialShape get_planar_output_shape(const ov::PartialShape& output_shape) const;
};

}  // namespace ov::snippets::op
