// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/iremote_context.hpp"

namespace ov {
namespace template_plugin {

template <class T>
class VectorTensorImpl;

// ! [vector_impl:implementation]
class VectorImpl : public ov::IRemoteTensor {
private:
    std::shared_ptr<ov::IRemoteTensor> m_tensor;

public:
    VectorImpl(const std::shared_ptr<ov::IRemoteTensor>& tensor) : m_tensor(tensor) {}

    template <class T>
    operator std::vector<T>&() const {
        auto impl = std::dynamic_pointer_cast<VectorTensorImpl<T>>(m_tensor);
        OPENVINO_ASSERT(impl, "Cannot get vector. Type is incorrect!");
        return impl->get();
    }

    void* get_data() {
        auto params = get_properties();
        OPENVINO_ASSERT(params.count("vector_data"), "Cannot get data. Tensor is incorrect!");
        try {
            auto* data = params.at("vector_data_ptr").as<void*>();
            return data;
        } catch (const std::bad_cast&) {
            OPENVINO_THROW("Cannot get data. Tensor is incorrect!");
        }
    }

    void set_shape(ov::Shape shape) override {
        m_tensor->set_shape(std::move(shape));
    }

    const ov::element::Type& get_element_type() const override {
        return m_tensor->get_element_type();
    }

    const ov::Shape& get_shape() const override {
        return m_tensor->get_shape();
    }

    size_t get_size() const override {
        return m_tensor->get_size();
    }

    size_t get_byte_size() const override {
        return m_tensor->get_byte_size();
    }

    const ov::Strides& get_strides() const override {
        return m_tensor->get_strides();
    }

    const ov::AnyMap& get_properties() const override {
        return m_tensor->get_properties();
    }

    const std::string& get_device_name() const override {
        return m_tensor->get_device_name();
    }
};
// ! [vector_impl:implementation]

}  // namespace template_plugin
}  // namespace ov
