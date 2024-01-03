// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ie_ngraph_utils.hpp"
#include "ie_remote_blob.hpp"
#include "ie_remote_context.hpp"
#include "openvino/runtime/iremote_context.hpp"

namespace ov {
namespace legacy_convert {

INFERENCE_ENGINE_API_CPP(ov::SoPtr<ov::IRemoteContext>)
convert_remote_context(const std::shared_ptr<InferenceEngine::RemoteContext>& context);
INFERENCE_ENGINE_API_CPP(InferenceEngine::Blob*) get_hardware_blob(InferenceEngine::Blob* blob);

class INFERENCE_ENGINE_API_CLASS(TensorHolder) {
public:
    TensorHolder(ov::SoPtr<ov::ITensor> tensor) : _tensor(tensor) {}

    const ov::SoPtr<ov::ITensor>& get_tensor() const {
        return _tensor;
    }

private:
    ov::SoPtr<ov::ITensor> _tensor;
};

}  // namespace legacy_convert

/**
 * @brief Tensor what contains InferenceEngine::RemoteBlob inside
 * Blob owns the memory
 */
class INFERENCE_ENGINE_API_CLASS(RemoteBlobTensor) : public IRemoteTensor {
    mutable element::Type m_type;
    mutable Shape m_shape;
    mutable Strides m_strides;
    mutable ov::AnyMap m_properties;
    mutable std::string m_dev_name;

public:
    std::shared_ptr<InferenceEngine::RemoteBlob> blob;

    RemoteBlobTensor(const InferenceEngine::RemoteBlob::Ptr& blob) : blob{blob} {
        OPENVINO_ASSERT(blob);
        m_shape = blob->getTensorDesc().getBlockingDesc().getBlockDims();
    }

    const element::Type& get_element_type() const override {
        m_type = InferenceEngine::details::convertPrecision(blob->getTensorDesc().getPrecision());
        return m_type;
    }

    void set_shape(ov::Shape shape) override {
        blob->setShape({shape.begin(), shape.end()});
    }

    const Shape& get_shape() const override {
        m_shape = blob->getTensorDesc().getBlockingDesc().getBlockDims();
        return m_shape;
    }

    const Strides& get_strides() const override {
        OPENVINO_ASSERT(get_element_type().bitwidth() >= 8,
                        "Could not get strides for types with bitwidths less then 8 bit. Tensor type: ",
                        get_element_type());
        const auto& element_strides = blob->getTensorDesc().getBlockingDesc().getStrides();
        const size_t elem_size = get_element_type().size();
        m_strides.clear();
        m_strides.resize(element_strides.size());
        std::transform(element_strides.begin(), element_strides.end(), m_strides.begin(), [&elem_size](size_t stride) {
            return stride * elem_size;
        });
        return m_strides;
    }

    size_t get_size() const override {
        return blob->size();
    }

    size_t get_byte_size() const override {
        return blob->byteSize();
    }

    const AnyMap& get_properties() const override {
        m_properties = blob->getParams();
        return m_properties;
    }

    const std::string& get_device_name() const override {
        m_dev_name = blob->getDeviceName();
        return m_dev_name;
    }
};

/**
 * @brief Create InferenceEngine::RemoteBlob from the Tensor
 */
class INFERENCE_ENGINE_API_CLASS(TensorRemoteBlob)
    : public InferenceEngine::RemoteBlob,
      public ov::legacy_convert::TensorHolder {
public:
    TensorRemoteBlob(const ov::SoPtr<ITensor>& tensor, InferenceEngine::TensorDesc desc)
        : InferenceEngine::RemoteBlob{desc},
          ov::legacy_convert::TensorHolder(tensor) {
        OPENVINO_ASSERT(this->get_tensor());
    }
    std::shared_ptr<ov::IRemoteTensor> cast_tensor() const {
        auto remote = std::dynamic_pointer_cast<ov::IRemoteTensor>(get_tensor()._ptr);
        OPENVINO_ASSERT(remote);
        return remote;
    }
    AnyMap getParams() const override {
        return cast_tensor()->get_properties();
    }
    std::string getDeviceName() const noexcept override {
        try {
            return cast_tensor()->get_device_name();
        } catch (...) {
            return {};
        }
    }
    std::shared_ptr<InferenceEngine::RemoteContext> getContext() const noexcept override {
        return {};
    }

    void allocate() noexcept override {}
    bool deallocate() noexcept override {
        return true;
    }
    InferenceEngine::LockedMemory<void> buffer() noexcept override {
        return {nullptr, nullptr, 0};
    }
    InferenceEngine::LockedMemory<const void> cbuffer() const noexcept override {
        return {nullptr, nullptr, 0};
    }
    InferenceEngine::LockedMemory<void> rwmap() noexcept override {
        return {nullptr, nullptr, 0};
    }
    InferenceEngine::LockedMemory<const void> rmap() const noexcept override {
        return {nullptr, nullptr, 0};
    }
    InferenceEngine::LockedMemory<void> wmap() noexcept override {
        return {nullptr, nullptr, 0};
    }
    const std::shared_ptr<InferenceEngine::IAllocator>& getAllocator() const noexcept override {
        return m_allocator;
    }
    void* getHandle() const noexcept override {
        return nullptr;
    }

    using TensorHolder::get_tensor;

private:
    std::shared_ptr<InferenceEngine::IAllocator> m_allocator;
};

}  // namespace ov

namespace InferenceEngine {

class INFERENCE_ENGINE_API_CLASS(IRemoteContextWrapper) : public ov::IRemoteContext {
private:
    std::shared_ptr<InferenceEngine::RemoteContext> m_context;
    mutable std::string m_name;
    mutable ov::AnyMap m_params;

public:
    IRemoteContextWrapper(const std::shared_ptr<InferenceEngine::RemoteContext>& context) : m_context(context) {}
    virtual ~IRemoteContextWrapper() = default;
    const std::shared_ptr<InferenceEngine::RemoteContext>& get_context();
    const std::string& get_device_name() const override;

    const ov::AnyMap& get_property() const override;

    ov::SoPtr<ov::IRemoteTensor> create_tensor(const ov::element::Type& type,
                                               const ov::Shape& shape,
                                               const ov::AnyMap& params = {}) override;
    ov::SoPtr<ov::ITensor> create_host_tensor(const ov::element::Type type, const ov::Shape& shape) override;
};

}  // namespace InferenceEngine
