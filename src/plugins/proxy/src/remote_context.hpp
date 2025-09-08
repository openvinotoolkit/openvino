// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>

#include "openvino/runtime/iremote_context.hpp"
#include "openvino/runtime/iremote_tensor.hpp"
#include "openvino/runtime/remote_context.hpp"
#include "openvino/runtime/so_ptr.hpp"

namespace ov {
namespace proxy {

/**
 * @brief Proxy remote context implementation
 * This class wraps hardware specific remote context and replace the context name
 */
class RemoteContext : public ov::IRemoteContext {
public:
    /**
     * @brief Constructs the proxy remote context
     *
     * @param ctx hardware context
     * @param dev_name device name without index
     * @param dev_index device index if exists else 0
     * @param has_index flag is true if device has an index and false in another case
     *
     * These arguments are needed to support the difference between legacy and 2.0 APIs.
     * In legacy API remote context doesn't contain the index in the name but Blob contains.
     * In 2.0 API Tensor and Context always contain device index
     */
    RemoteContext(ov::SoPtr<ov::IRemoteContext>&& ctx, const std::string& dev_name, size_t dev_index, bool has_index);

    RemoteContext(const ov::SoPtr<ov::IRemoteContext>& ctx,
                  const std::string& dev_name,
                  size_t dev_index,
                  bool has_index);
    const std::string& get_device_name() const override;

    const ov::AnyMap& get_property() const override;

    ov::SoPtr<ov::IRemoteTensor> create_tensor(const ov::element::Type& type,
                                               const ov::Shape& shape,
                                               const ov::AnyMap& params = {}) override;

    ov::SoPtr<ov::ITensor> create_host_tensor(const ov::element::Type type, const ov::Shape& shape) override;

    ov::SoPtr<ov::ITensor> wrap_tensor(const ov::SoPtr<ov::ITensor>& tensor);

    static const ov::SoPtr<ov::IRemoteContext>& get_hardware_context(const ov::SoPtr<ov::IRemoteContext>& context);

private:
    ov::SoPtr<ov::IRemoteContext> m_context;
    std::string m_name;

    void init_context(const std::string& dev_name, size_t dev_index, bool has_index);
};

}  // namespace proxy
}  // namespace ov
