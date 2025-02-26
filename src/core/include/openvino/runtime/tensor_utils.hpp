// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <filesystem>

#include "openvino/core/partial_shape.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {

/// \brief Save given tensor data into a file. File will contain only raw bytes of a tensor.data as it is allocated in
///        memory. No element type nor shape nor other metadata are serialized. Strides are preserved.
/// \param tensor Tensor which data will be serialized.
/// \param file_name Path to the output file
OPENVINO_API
void save_tensor_data(const Tensor& tensor, const std::filesystem::path& file_name);

/// \brief Read a tensor content from a file. Only raw data is loaded.
/// \param file_name Path to file to read.
/// \param element_type Element type, when not specified the it is assumed as element::u8.
/// \param shape Shape for resulting tensor. If provided shape is static, specified number of elements is read only.
///              File should contain enough bytes, an exception is raised otherwise.
///              One of the dimensions can be dynamic. In this case it will be determined automatically based on the
///              length of the file content and `offset`. Default value is [?].
/// \param offset Read file starting from specified offset. Default is 0. The remining size of the file should be
/// compatible with shape.
/// \param mmap Use mmap that postpones real read from file until data is accessed. If mmap is used, the file
///             should not be modified until returned tensor is destroyed.
OPENVINO_API
Tensor read_tensor_data(const std::filesystem::path& file_name,
                        const element::Type& element_type = element::u8,
                        const PartialShape& shape = PartialShape{Dimension::dynamic()},
                        std::size_t offset = 0,
                        bool mmap = true);

/// \brief Read raw data from a file into pre-allocated tensor.
/// \param file_name Path to the input file with raw tensor data.
/// \param tensor Tensor to read data to. Tensor should have correct element_type and shape set that is used to
/// determine how many bytes will be read from the file.
/// \param offset Read file starting from specified offset. Default
/// is 0. The remining part of the file should contain enough bytes to satisfy tensor size.
OPENVINO_API
void read_tensor_data(const std::filesystem::path& file_name, Tensor& tensor, std::size_t offset = 0);

/// \brief Save given tensor data into a temporary file. Read the content from the file to a new tensor using mmap.
///        The temporary file is removed when the returned tensor and all its copies are destroyed.
/// \param tensor Tensor to read data to. Tensor should have correct element_type and shape set that is used to
/// \param file_name Path to the temporary file

OPENVINO_API
Tensor create_mmaped_tensor(const Tensor& tensor, const std::filesystem::path& file_name);

}  // namespace ov
