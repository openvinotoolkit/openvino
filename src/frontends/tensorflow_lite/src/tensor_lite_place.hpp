// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/frontend.hpp"
#include "openvino/frontend/tensorflow_lite/visibility.hpp"
#include "openvino/frontend/tensorflow/decoder.hpp"
#include "utils.hpp"
#include "place.hpp"

namespace ov {
namespace frontend {
namespace tensorflow_lite {

struct Quantization {
    std::vector<float> scale;
    std::vector<int64_t> zero_point;
    int64_t axis{};
    bool no_quantization = true;
};

class TENSORFLOW_LITE_API TensorLitePlace : public ov::frontend::tensorflow::TensorPlace {
public:
    TensorLitePlace(const ov::frontend::InputModel& input_model,
                    const ov::PartialShape& pshape,
                    ov::element::Type type,
                    const std::vector<std::string>& names,
                    ov::frontend::tensorflow_lite::Quantization quantization) : ov::frontend::tensorflow::TensorPlace(input_model, pshape, type, names),
                                             m_quantization(quantization) {};

    const struct ov::frontend::tensorflow_lite::Quantization& get_quantization() const;
protected:
    ov::frontend::tensorflow_lite::Quantization m_quantization;
};

}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
