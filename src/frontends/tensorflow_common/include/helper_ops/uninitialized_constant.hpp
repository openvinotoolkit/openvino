// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>

#include "internal_operation.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {

class UninitializedConstant : public InternalOperation {
public:
    OPENVINO_OP("UninitializedConstant", "ov::frontend::tensorflow::util", InternalOperation);

    UninitializedConstant(const std::shared_ptr<DecoderBase>& decoder = std::make_shared<DecoderFake>())
        : InternalOperation(decoder, {}, 1) {
        validate_and_infer_types();
    }

    UninitializedConstant(const element::Type& type, const Shape& shape)
        : InternalOperation(std::make_shared<DecoderFake>(), {}, 1),
          m_element_type(type),
          m_shape(shape) {
        set_output_type(0, type, shape);
    }

    void validate_and_infer_types() override {
        set_output_type(0, ov::element::undefined, ov::PartialShape::dynamic());
    }

private:
    element::Type m_element_type;
    Shape m_shape{};
};

}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
