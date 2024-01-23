// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "helper_ops/internal_operation.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {

class HashTable : public InternalOperation {
public:
    OPENVINO_OP("HashTable", "ov::frontend::tensorflow", InternalOperation);

    HashTable(const std::shared_ptr<DecoderBase>& decoder = nullptr)
        : InternalOperation(decoder, OutputVector{}, 1, "HashTable") {
        validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        set_output_type(0, ov::element::dynamic, ov::PartialShape::dynamic());
    }

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override {
        auto hash_table_node = std::make_shared<HashTable>(m_decoder);
        hash_table_node->set_attrs(get_attrs());
        return hash_table_node;
    }
};

}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
