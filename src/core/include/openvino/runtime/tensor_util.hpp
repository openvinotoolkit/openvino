// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/partial_shape.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {

/// \brief Save given tensor data into a file. File will contain only raw bytes of a tensor.data as it is allocated in
/// memory.
///        No element type nor shape nor other metadata are serialized. Strides are preserved.
/// \param tensor Tensor which data will be serialized.
/// \param file_name Path to the output file
OPENVINO_API
void save_tensor_data(const Tensor& tensor, const std::string& file_name);

#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT)
OPENVINO_API
void save_tensor_data(const Tensor& tensor, const std::wstring& output_model);
#endif

/// \brief Read a tensor content from a file. Only raw data is loaded.
/// \param file_name Path to the output file
/// \param element_type Element type, when not specified the it is assumed as element::u8.
/// \param shape Shape for resulting tensor. If provided shape is static, specified number of elements is read only.
/// File should contain enough bytes, an exception is raised otherwise.
///              One of the dimensions can be dynamic. In this case it will be determined automatically based on the
///              length of the file content and `offset`. Default value is [?].
/// \param offset Read file starting from specified offset. Default is 0. The remining size of the file should be
/// compatible with shape. \param mmap Use mmap that postpones real read from file until data is accessed.
OPENVINO_API
Tensor read_tensor_data(const std::string& file_name,
                        const element::Type& element_type = element::u8,
                        const PartialShape& shape = PartialShape{Dimension::dynamic()},
                        std::size_t offset = 0,
                        bool mmap = true);

#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT)
OPENVINO_API
Tensor read_tensor_data(const std::wstring& file_name,
                        const element::Type& element_type = element::u8,
                        const PartialShape& shape = PartialShape{Dimension::dynamic()},
                        std::size_t offset = 0,
                        bool mmap = true);
#endif

/// \brief Read raw data from a file into pre-allocated tensor.
/// \param file_name Path to the input file with raw tensor data.
/// \param tensor Tensor to read data to. Tensor should have correct element_type and shape set that is used to
/// determine how many bytes will be read from the file. \param offset Read file starting from specified offset. Default
/// is 0. The remining part of the file should contain enough bytes to satisfy tensor size.
OPENVINO_API
void read_tensor_data(const std::string& file_name, Tensor& tensor, std::size_t offset = 0);

#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT)
OPENVINO_API
void read_tensor_data(const std::wstring& file_name, Tensor& tensor, std::size_t offset = 0);
#endif

/// \brief Read raw data from a file into a tensor. Optionally re-allocate memory in tensor if required.
/// \param file_name Path to the input file with raw tensor data.
/// \param tensor Tensor to read data to. Memory is allocated using set_shape method.
/// \param shape Shape for resulting tensor. If provided shape is static, specified number of elements is read only.
/// File should contain enough bytes, an exception is raised otherwise.
///              One of the dimensions can be dynamic. In this case it will be determined automatically based on the
///              length of the file content and `offset`.
/// \param offset Read file starting from specified offset. Default is 0. The remining size of the file should be
/// compatible with shape.
OPENVINO_API
void read_tensor_data(const std::string& file_name, Tensor& tensor, const PartialShape& shape, std::size_t offset = 0);

#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT)
OPENVINO_API
void read_tensor_data(const std::wstring& file_name, Tensor& tensor, const PartialShape& shape, std::size_t offset = 0);
#endif

}  // namespace ov