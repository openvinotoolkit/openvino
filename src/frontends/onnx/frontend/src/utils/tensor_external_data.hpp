// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <onnx/onnx_pb.h>

#include "openvino/runtime/aligned_buffer.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "openvino/util/mmap_object.hpp"

namespace ngraph {
namespace onnx_import {
namespace detail {
template <class T>
using Buffer = std::shared_ptr<ov::SharedBuffer<std::shared_ptr<T>>>;
using MappedMemoryHandles = std::shared_ptr<std::map<std::string, std::shared_ptr<ov::MappedMemory>>>;
/// \brief  Helper class used to load tensor data from external files
class TensorExternalData {
public:
    TensorExternalData(const ONNX_NAMESPACE::TensorProto& tensor);

    /// \brief      Load external data from tensor passed to constructor
    ///
    /// \note       If read data from external file fails,
    /// \note       If reading data from external files fails,
    ///             the invalid_external_data exception is thrown.
    ///
    /// \return     External binary data loaded into the SharedBuffer
    Buffer<ov::AlignedBuffer> load_external_data(const std::string& model_dir) const;

    /// \brief      Map (mmap for lin, MapViewOfFile for win) external data from tensor passed to constructor
    ///
    /// \note       If read data from external file fails,
    /// \note       If reading data from external files fails,
    ///             the invalid_external_data exception is thrown.
    ///
    /// \return     External binary data loaded into the SharedBuffer
    Buffer<ov::MappedMemory> load_external_mmap_data(const std::string& model_dir, MappedMemoryHandles cache) const;

    /// \brief      Represets parameter of external data as string
    ///
    /// \return     State of TensorExternalData as string representation
    std::string to_string() const;

private:
    std::string m_data_location{};
    uint64_t m_offset = 0;
    uint64_t m_data_length = 0;
    std::string m_sha1_digest{};
};
}  // namespace detail
}  // namespace onnx_import
}  // namespace ngraph
