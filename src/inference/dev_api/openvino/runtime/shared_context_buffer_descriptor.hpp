// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>

#include "openvino/runtime/aligned_buffer.hpp"
#include "openvino/runtime/iremote_context.hpp"
#include "openvino/runtime/iremote_tensor.hpp"
#include "openvino/runtime/so_ptr.hpp"

namespace ov {

/// \brief Buffer descriptor that carries the set of remote contexts which co-own
///        (or should receive) the underlying allocation.
///
/// Extends the base IBufferDescriptor interface so that consumers can
/// retrieve the associated remote contexts via get_remote_contexts().

class OPENVINO_RUNTIME_API SharedContextBufferDescriptor : public ov::IBufferDescriptor {
public:
    struct RemoteContextPtrLess {
        bool operator()(const ov::SoPtr<ov::IRemoteContext>& lhs,
                        const ov::SoPtr<ov::IRemoteContext>& rhs) const noexcept {
            return std::less<const ov::IRemoteContext*>{}(lhs._ptr.get(), rhs._ptr.get());
        }
    };

    using RemoteContextsMap = std::map<ov::SoPtr<ov::IRemoteContext>,
                                       ov::SoPtr<ov::IRemoteTensor>,
                                       RemoteContextPtrLess>;

    SharedContextBufferDescriptor(size_t id,
                                  size_t offset,
                                  size_t real_buffer_size,
                                  const std::shared_ptr<ov::AlignedBuffer>& source_buffer,
                                  RemoteContextsMap remote_contexts);

    size_t get_id() const override;
    size_t get_offset() const override;
    size_t get_real_buffer_size() const;
    std::shared_ptr<ov::AlignedBuffer> get_source_buffer() const override;

    const RemoteContextsMap& get_remote_contexts() const;

    ~SharedContextBufferDescriptor() override;

private:
    size_t m_id = 0;
    size_t m_offset = 0;
    size_t m_real_buffer_size = 0;
    std::weak_ptr<ov::AlignedBuffer> m_source_buffer;
    RemoteContextsMap m_remote_contexts;
};

}  // namespace ov
