// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <string>

namespace ov::frontend::onnx {

/// \brief Contents of an external data file supplied by an ExternalDataReader.
struct ExternalDataBuffer {
    const void* data = nullptr;   ///< Start of the file contents, or nullptr if the reader doesn't supply the file
    size_t size = 0;              ///< Size of the file contents in bytes
    std::shared_ptr<void> owner;  ///< Keeps `data` valid; held for as long as constants refer to the data
};

/// \brief Supplies the contents of an external data file declared by an ONNX model.
///
/// The frontend calls it at most once per file per model, the first time an initializer refers to the file, with the
/// file's `location` as declared in the model's `external_data` entries, relative to the model directory (normalized,
/// with '/' separators). The returned buffer must hold the whole file; initializers are views into it at their
/// declared offsets, so no data is copied. Returning a buffer with `data == nullptr` means the reader doesn't supply
/// the file, and the frontend reads it from the file system. Exceptions are reported as invalid external data.
///
/// Pass it to FrontEnd::load() as an entry of an ov::AnyMap load parameter, under the key `external_data_reader`.
using ExternalDataReader = std::function<ExternalDataBuffer(const std::string& location)>;

/// \brief ov::AnyMap key of the ExternalDataReader load parameter.
constexpr const char* external_data_reader = "ONNX_EXTERNAL_DATA_READER";

}  // namespace ov::frontend::onnx
