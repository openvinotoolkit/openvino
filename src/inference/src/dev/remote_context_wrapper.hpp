// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "dev/make_tensor.hpp"
#include "ie_ngraph_utils.hpp"
#include "ie_remote_context.hpp"
#include "openvino/runtime/iremote_context.hpp"

namespace ov {

class RemoteContextWrapper : public InferenceEngine::RemoteContext {
private:
    std::shared_ptr<ov::IRemoteContext> m_context;

public:
    RemoteContextWrapper(const std::shared_ptr<ov::IRemoteContext>& context) : m_context(context) {}

    const std::shared_ptr<ov::IRemoteContext>& get_context() const {
        return m_context;
    }

    std::string getDeviceName() const noexcept override {
        return m_context->get_device_name();
    }

    InferenceEngine::RemoteBlob::Ptr CreateBlob(const InferenceEngine::TensorDesc& tensorDesc,
                                                const InferenceEngine::ParamMap& params = {}) override {
        return std::dynamic_pointer_cast<InferenceEngine::RemoteBlob>(ov::tensor_to_blob(
            m_context->create_tensor(InferenceEngine::details::convertPrecision(tensorDesc.getPrecision()),
                                     tensorDesc.getBlockingDesc().getBlockDims(),
                                     params),
            false));
    }

    InferenceEngine::MemoryBlob::Ptr CreateHostBlob(const InferenceEngine::TensorDesc& tensorDesc) override {
        return std::dynamic_pointer_cast<InferenceEngine::MemoryBlob>(ov::tensor_to_blob(
            m_context->create_host_tensor(InferenceEngine::details::convertPrecision(tensorDesc.getPrecision()),
                                          tensorDesc.getBlockingDesc().getBlockDims()),
            false));
    }

    InferenceEngine::ParamMap getParams() const override {
        return m_context->get_property();
    }
};

}  // namespace ov
