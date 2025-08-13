// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <onnx/onnx_pb.h>

#include "openvino/runtime/aligned_buffer.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "openvino/util/mmap_object.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace detail {
using ::ONNX_NAMESPACE::TensorProto;
template <class T>
using Buffer = std::shared_ptr<ov::SharedBuffer<std::shared_ptr<T>>>;
using MappedMemoryHandles = std::shared_ptr<std::map<std::string, std::shared_ptr<ov::MappedMemory>>>;
/// \brief  Helper class used to load tensor data from external files
class TensorExternalData {
public:
    TensorExternalData(const TensorProto& tensor);

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

    /// \brief      Object contains a data length after construction. Method allows read-only access to this
    ///             information.
    ///
    /// \return     Returns a stored data size in bytes
    uint64_t size() const {
        return m_data_length;
    }

private:
    std::string m_data_location{};
    uint64_t m_offset = 0;
    uint64_t m_data_length = 0;
    std::string m_sha1_digest{};
};

/*
As
https://github.com/microsoft/onnxruntime/blob/4f6ae14e09729b3e3aba921de2e5bcc26d3e7768/onnxruntime/core/framework/tensorprotoutils.h#L206
describes, this is a special marker used to indicate the external weights is already in the shared memory from ORT. if
location field is set to this marker, the offset field contain the address of the memory.
*/
const std::string ORT_MEM_ADDR = "*/_ORT_MEM_ADDR_/*";

class MappedMemoryHolder : public ov::MappedMemory {
public:
    MappedMemoryHolder(char* addr_ptr, size_t length) : m_size(length) {
        m_copied_data = std::make_shared<ov::AlignedBuffer>(length);
        std::memcpy(m_copied_data->get_ptr<char>(), addr_ptr, length);
    }

    ~MappedMemoryHolder() {}

    char* data() noexcept override {
        return m_copied_data->get_ptr<char>();
    }

    size_t size() const noexcept override {
        return m_size;
    }

private:
    std::shared_ptr<ov::AlignedBuffer> m_copied_data;
    size_t m_size = 0;
};
}  // namespace detail
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
