// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>

#include "openvino/runtime/iremote_context.hpp"
#include "openvino/runtime/iremote_tensor.hpp"
#include "openvino/runtime/remote_context.hpp"

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
     * @param is_new_api flag reports which API is used
     *
     * These arguments are needed to support the difference between legacy and 2.0 APIs.
     * In legacy API remote context doesn't contain the index in the name but Blob contains.
     * In 2.0 API Tensor and Context always contain device index
     */
    RemoteContext(ov::RemoteContext&& ctx,
                  const std::string& dev_name,
                  size_t dev_index,
                  bool has_index,
                  bool is_new_api);
    const std::string& get_device_name() const override;

    const ov::AnyMap& get_property() const override;

    std::shared_ptr<ov::IRemoteTensor> create_tensor(const ov::element::Type& type,
                                                     const ov::Shape& shape,
                                                     const ov::AnyMap& params = {}) override;

    std::shared_ptr<ov::ITensor> create_host_tensor(const ov::element::Type type, const ov::Shape& shape) override;

    ov::Tensor wrap_tensor(const ov::RemoteTensor& tensor);

    static const ov::RemoteContext& get_hardware_context(const ov::RemoteContext& context);
    static const std::shared_ptr<ov::IRemoteContext>& get_hardware_context(
        const std::shared_ptr<ov::IRemoteContext>& context);

private:
    ov::RemoteContext m_context;
    std::string m_name;
    std::string m_tensor_name;

    std::string get_tensor_name() const;
};

}  // namespace proxy
}  // namespace ov
