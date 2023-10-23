// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "internal_operation.hpp"
#include "merge.hpp"
#include "tf_utils.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {

// Internal operation for NextIteration that makes its input available to the next iteration
// the output is going to Merge node.
class NextIteration : public InternalOperation {
public:
    OPENVINO_OP("NextIteration", "ov::frontend::tensorflow", InternalOperation);

    NextIteration(const std::shared_ptr<DecoderBase>& decoder = std::make_shared<DecoderFake>())
        : InternalOperation(decoder, OutputVector{}, 1, "NextIteration"),
          m_back_edge_set(false) {
        validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        set_output_type(0, ov::element::dynamic, ov::PartialShape::dynamic());
    }

    void set_producer(const std::string& producer_name, size_t producer_output_port_idx) {
        m_producer_name = producer_name;
        m_producer_output_port_idx = producer_output_port_idx;
        m_back_edge_set = true;
    }

    void get_producer(std::string& producer_name, size_t& producer_output_port_idx) const {
        FRONT_END_GENERAL_CHECK(m_back_edge_set,
                                "[TensorFlow Frontend] internal error: back edge for NextIteration is not set");
        producer_name = m_producer_name;
        producer_output_port_idx = m_producer_output_port_idx;
    }

private:
    bool m_back_edge_set;
    std::string m_producer_name;
    size_t m_producer_output_port_idx;
};

}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
