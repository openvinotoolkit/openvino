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

/// Pseudo-entity for storing strings
class StringConstant : public InternalOperation {
public:
    OPENVINO_OP("StringConstant", "ov::frontend::tensorflow::util", InternalOperation);

    StringConstant(ov::Any data, const std::shared_ptr<DecoderBase>& decoder = std::make_shared<DecoderFake>())
        : InternalOperation(decoder, {}, 1),
          m_data(data) {
        validate_and_infer_types();
    }

    StringConstant(std::string& str, const std::shared_ptr<DecoderBase>& decoder = std::make_shared<DecoderFake>())
        : InternalOperation(decoder, {}, 1),
          m_data({str}) {
        validate_and_infer_types();
    }

    StringConstant(const std::shared_ptr<DecoderBase>& decoder = std::make_shared<DecoderFake>())
        : InternalOperation(decoder, {}, 1) {
        validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        set_output_type(0, ov::element::dynamic, ov::PartialShape::dynamic());
    }

    ov::Any get_data() {
        return m_data;
    }

    std::string& get_string() {
        return m_data.as<std::vector<std::string>>()[0];
    }

private:
    ov::Any m_data;
    ov::Shape m_shape;
};

}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
