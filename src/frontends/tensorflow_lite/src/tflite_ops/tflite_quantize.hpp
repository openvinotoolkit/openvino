// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>

#include "helper_ops/internal_operation.hpp"
#include "openvino/frontend/decoder.hpp"
#include "openvino/frontend/tensorflow_lite/quantization_info.hpp"

namespace ov {
namespace frontend {
namespace tensorflow_lite {

class TFLQuantize : public ov::frontend::tensorflow::InternalOperation {
public:
    OPENVINO_OP("TFLQuantize", "ov::frontend::tensorflow_lite::util", ov::frontend::tensorflow::InternalOperation);

    TFLQuantize(const Output<Node>& data,
                std::shared_ptr<ov::frontend::tensorflow_lite::QuantizationInfo> info,
                const element::Type& type,
                const std::shared_ptr<DecoderBase>& decoder = nullptr)
        : ov::frontend::tensorflow::InternalOperation(decoder, OutputVector{data}, 1, "TFLQuantize"),
          m_info(info),
          m_type(type),
          m_original_type(type) {
        validate_and_infer_types();
    }

    void set_type(const element::Type& type) {
        m_type = type;
    }

    element::Type get_type() const {
        return m_type;
    }

    element::Type get_original_type() const {
        return m_original_type;
    }

    std::shared_ptr<QuantizationInfo> get_info() const {
        return m_info;
    }

    void validate_and_infer_types() override {
        set_output_type(0, m_type, get_input_partial_shape(0));
    }

private:
    std::shared_ptr<ov::frontend::tensorflow_lite::QuantizationInfo> m_info;
    ov::element::Type m_type;
    const ov::element::Type m_original_type;
};
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
