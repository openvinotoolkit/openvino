// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <onnx/onnx_pb.h>

#include <fstream>
#include <string>

namespace ov {
namespace frontend {
namespace onnx {
namespace common {
using namespace ::ONNX_NAMESPACE;
/// \brief   Parses an ONNX model from a file located on a storage device.
///
/// \param   file_path    Path to the file containing an ONNX model.
///
/// \return  The parsed in-memory representation of the ONNX model
ModelProto parse_from_file(const std::string& file_path);
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
ModelProto parse_from_file(const std::wstring& file_path);
#endif

/// \brief   Parses an ONNX model from a stream (representing for example a file)
///
/// \param   model_stream  Path to the file containing an ONNX model.
///
/// \return  The parsed in-memory representation of the ONNX model
ModelProto parse_from_istream(std::istream& model_stream);
}  // namespace common
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
