// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>

#include "internal_operation.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {

class UnsupportedConstant : public InternalOperation {
public:
    OPENVINO_OP("UnsupportedConstant", "ov::frontend::tensorflow::util", InternalOperation);

    UnsupportedConstant(const std::shared_ptr<DecoderBase>& decoder = std::make_shared<DecoderFake>())
        : InternalOperation(decoder, {}, 1, "Const of unknown type") {
        validate_and_infer_types();
    }

    UnsupportedConstant(const std::string& no_conversion_reason,
                        const std::shared_ptr<DecoderBase>& decoder = std::make_shared<DecoderFake>())
        : InternalOperation(decoder, {}, 1, no_conversion_reason) {
        validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        set_output_type(0, ov::element::dynamic, ov::PartialShape::dynamic());
    }
};

}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
