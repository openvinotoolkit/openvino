// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "helper_ops/internal_op.hpp"
#include "openvino/core/validation_util.hpp"

namespace ov::frontend::pytorch {

class BatchedTensor : public InternalOperation {
public:
    OPENVINO_OP("BatchedTensor", "internal", InternalOperation);

    BatchedTensor(const Output<Node>& value, int64_t batch_dim, int64_t level)
        : InternalOperation("BatchedTensor", {value}, 1, "Batched tensor was not removed by its vmap scope."),
          m_batch_dim(batch_dim),
          m_level(level) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        const auto& shape = get_input_partial_shape(0);
        if (shape.rank().is_dynamic()) {
            set_output_type(0, get_input_element_type(0), PartialShape::dynamic());
            return;
        }
        std::vector<Dimension> dimensions(shape.begin(), shape.end());
        NODE_VALIDATION_CHECK(this, m_batch_dim >= 0 && m_batch_dim < shape.rank().get_length(), "Invalid batch axis.");
        dimensions.erase(dimensions.begin() + m_batch_dim);
        set_output_type(0, get_input_element_type(0), PartialShape(dimensions));
    }

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override {
        check_new_args_count(this, inputs);
        return std::make_shared<BatchedTensor>(inputs.at(0), m_batch_dim, m_level);
    }

    int64_t get_batch_dim() const {
        return m_batch_dim;
    }

    int64_t get_level() const {
        return m_level;
    }

private:
    int64_t m_batch_dim;
    int64_t m_level;
};

void check_vmap_node(const NodeContext& context);

}  // namespace ov::frontend::pytorch
