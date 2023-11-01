// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "internal_operation.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {

// Internal operation for TensorArrayV3
// An array of Tensors of given size
// It has two outputs:
// 1. handle - resource (a reference) for tensor array
// 2. flow_out - float type will be used for storing tensor array
class TensorArrayV3 : public InternalOperation {
public:
    OPENVINO_OP("TensorArrayV3", "ov::frontend::tensorflow", InternalOperation);

    TensorArrayV3(const Output<Node>& size,
                  const ov::element::Type element_type,
                  const std::shared_ptr<DecoderBase>& decoder = std::make_shared<DecoderFake>())
        : InternalOperation(decoder, OutputVector{size}, 2, "TensorArrayV3"),
          m_element_type(element_type),
          m_element_rank(-1) {
        validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        set_output_type(0, m_element_type, ov::PartialShape::dynamic());
        set_output_type(1, m_element_type, ov::PartialShape::dynamic());
    }

    ov::element::Type get_element_type() const {
        return m_element_type;
    }

    int64_t get_element_rank() const {
        return m_element_rank;
    }

    void set_element_rank(int64_t element_rank) {
        FRONT_END_GENERAL_CHECK(
            element_rank >= 0,
            "[TensorFlow Frontend] internal error: negavite element rank tries to set for TensorArrayV3");
        m_element_rank = element_rank;
    }

private:
    ov::element::Type m_element_type;
    int64_t m_element_rank;
};

}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
