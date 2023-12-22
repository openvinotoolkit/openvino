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
};

}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
