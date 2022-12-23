// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "schema_generated.h"
#include "place.hpp"
#include "decoder_flatbuffer.h"

namespace ov {
namespace frontend {
namespace tensorflow_lite {

class TensorLitePlace;
struct Quantization;

ov::element::Type get_ov_type(const tflite::TensorType& tf_type);

ov::PartialShape get_ov_shape(const flatbuffers::Vector<int32_t>* tf_shape);

//ov::frontend::tensorflow_lite::Quantization get_quantization(const tflite::QuantizationParameters* tf_quantization);
std::shared_ptr<ov::frontend::tensorflow_lite::Quantization> get_quantization(const tflite::QuantizationParameters* tf_quantization);

ov::Output<Node> apply_quantization(ov::Output<ov::Node> output,
                                    const std::shared_ptr<ov::frontend::tensorflow::TensorPlace>& tensor, bool is_input=false);

}
}
}