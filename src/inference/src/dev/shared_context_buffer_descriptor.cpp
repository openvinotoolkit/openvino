// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/shared_context_buffer_descriptor.hpp"

#include <utility>

namespace ov {

SharedContextBufferDescriptor::SharedContextBufferDescriptor(
    size_t id,
    size_t offset,
    size_t real_buffer_size,
    const std::shared_ptr<ov::AlignedBuffer>& source_buffer,
    RemoteContextsMap remote_contexts)
    : m_id(id),
      m_offset(offset),
      m_real_buffer_size(real_buffer_size),
      m_source_buffer(source_buffer),
      m_remote_contexts(std::move(remote_contexts)) {}

SharedContextBufferDescriptor::~SharedContextBufferDescriptor() = default;

size_t SharedContextBufferDescriptor::get_id() const {
    return m_id;
}

size_t SharedContextBufferDescriptor::get_offset() const {
    return m_offset;
}

size_t SharedContextBufferDescriptor::get_real_buffer_size() const {
    return m_real_buffer_size;
}

std::shared_ptr<ov::AlignedBuffer> SharedContextBufferDescriptor::get_source_buffer() const {
    return m_source_buffer.lock();
}

const SharedContextBufferDescriptor::RemoteContextsMap& SharedContextBufferDescriptor::get_remote_contexts() const {
    return m_remote_contexts;
}

}  // namespace ov
