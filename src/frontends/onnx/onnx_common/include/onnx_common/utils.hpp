// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <onnx/onnx_pb.h>

#include "openvino/core/partial_shape.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace common {
using ::ONNX_NAMESPACE::TensorProto_DataType;
using ::ONNX_NAMESPACE::TensorShapeProto;

/// \brief Retuns size of an ONNX data type in bytes.
///
/// \param onnx_type Number assigned to an ONNX data type in the TensorProto_DataType enum.
///
size_t get_onnx_data_size(int32_t onnx_type);

/// \brief Retuns a OpenVINO data type corresponding to an ONNX type.
///
/// \param onnx_type An element of TensorProto_DataType enum which determines an ONNX type.
///
ov::element::Type_t onnx_to_ov_data_type(const TensorProto_DataType& onnx_type);

/// \brief Retuns an ONNX data type corresponding to a OpenVINO data type.
///
/// \param ov_type An element of  ov::element::Type_t enum class which determines a OpenVINO data
/// type.
///
TensorProto_DataType ov_to_onnx_data_type(const ov::element::Type_t& ov_type);

/// \brief Retuns true if a OpenVINO data type is mapped to an ONNX data type.
///
/// \param ov_type An element of  ov::element::Type_t enum class which determines a OpenVINO data
/// type.
///
bool is_supported_ov_type(const ov::element::Type_t& ov_type);

/// \brief Retuns OpenVINO PartialShape based on onnx_shape.
///
/// \param onnx_shape A shape of tensor represented in ONNX way.
///
PartialShape onnx_to_ov_shape(const TensorShapeProto& onnx_shape);

}  // namespace common
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
