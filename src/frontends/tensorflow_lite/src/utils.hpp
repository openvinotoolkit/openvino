// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "decoder_flatbuffer.h"
#include "openvino/frontend/tensorflow_lite/quantization_info.hpp"
#include "openvino/frontend/tensorflow_lite/sparsity_info.hpp"
#include "place.hpp"
#include "schema_generated.h"

namespace ov {
namespace frontend {
namespace tensorflow_lite {

class TensorLitePlace;

ov::element::Type get_ov_type(const tflite::TensorType& tf_type);
ov::PartialShape get_ov_shape(const flatbuffers::Vector<int32_t>* tf_shape,
                              const flatbuffers::Vector<int32_t>* tf_shape_sig);
std::shared_ptr<QuantizationInfo> get_quantization(const tflite::QuantizationParameters* tf_quantization);
void apply_quantization(ov::Output<ov::Node>& output, ov::element::Type type);
void dequantize_inputs(OutputVector& deq_inputs);
std::shared_ptr<SparsityInfo> get_sparsity(const flatbuffers::Vector<int32_t>* tf_shape,
                                           const tflite::SparsityParameters* tf_sparsity,
                                           const ov::element::Type target_type,
                                           const uint8_t* buffer);

template <typename T>
OutputVector get_indexed_outputs(const T& outputs);

}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov