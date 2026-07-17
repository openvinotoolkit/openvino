// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <vector>

#include "openvino/runtime/aligned_buffer.hpp"
#include "openvino/runtime/iremote_context.hpp"
#include "openvino/runtime/shared_context_buffer_descriptor.hpp"
#include "openvino/runtime/so_ptr.hpp"

namespace ov {
namespace npuw {
class SharedContextBuffer : public ov::AlignedBuffer, public std::enable_shared_from_this<SharedContextBuffer> {
public:
    using RemoteContextsMap = SharedContextBufferDescriptor::RemoteContextsMap;
    SharedContextBuffer(size_t byte_size,
                        std::vector<ov::SoPtr<ov::IRemoteContext>> remote_contexts,
                        size_t alignment = 64);

    ~SharedContextBuffer() override;

    std::shared_ptr<ov::IBufferDescriptor> get_descriptor() const override;

    ov::SoPtr<ov::IRemoteTensor> get_remote_tensors_if_exist(ov::SoPtr<ov::IRemoteContext> remote_ctx) const;

    size_t get_real_buffer_size() const;
private:
    size_t m_id = 0;
    RemoteContextsMap m_remote_contexts;
    size_t m_real_buffer_size = 0;
    ov::SoPtr<ov::IRemoteTensor> bind_remote_context(const ov::SoPtr<ov::IRemoteContext>& remote_ctx);
};
}  // namespace npuw
}  // namespace ov
