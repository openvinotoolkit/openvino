// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <map>
#include <string>

#include "decoder_map.hpp"
#include "op_table.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/frontend/tensorflow/node_context.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino_conversions.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace tensorflow_lite {

class QuantizationInfo : public ov::RuntimeAttribute {
public:
    OPENVINO_RTTI("QuantizationInfo");
    QuantizationInfo() = default;
    explicit QuantizationInfo(std::shared_ptr<Quantization> quantization) {
        m_quantization = quantization;
    }
    bool is_copyable() const override {
        return false;
    }
private:
    std::shared_ptr<Quantization> m_quantization;
};
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
