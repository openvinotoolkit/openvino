// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "remote_context.hpp"

#include <memory>

#include "openvino/core/any.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type_traits.hpp"
#include "openvino/runtime/itensor.hpp"
#include "template/remote_tensor.hpp"

namespace {

template <class T>
class VectorTensorImpl : public ov::ITensor {
public:
    VectorTensorImpl(const ov::element::Type element_type, const ov::Shape& shape)
        : m_element_type{element_type},
          m_shape{shape},
          m_data(ov::shape_size(shape)) {}

    const ov::element::Type& get_element_type() const override {
        return m_element_type;
    }

    const ov::Shape& get_shape() const override {
        return m_shape;
    }

    void set_shape(const ov::Shape& new_shape) override {
        auto old_byte_size = get_byte_size();
        OPENVINO_ASSERT(shape_size(new_shape) * get_element_type().size() <= old_byte_size,
                        "Could set new shape: ",
                        new_shape);
        m_shape = new_shape;
    }

    ov::AnyMap get_properties() const override {
        return {{ov::device::id.name(), "TEMPLATE"}};
    }

    std::vector<T> get() {
        return m_data;
    }

    void* data(const ov::element::Type& type = {}) const override {
        OPENVINO_NOT_IMPLEMENTED;
    }

protected:
    ov::element::Type m_element_type;
    ov::Shape m_shape;
    std::vector<T> m_data;
};

}  // namespace

namespace ov {
namespace template_plugin {

class VectorTensor::VectorImpl : public ov::ITensor {
private:
    std::shared_ptr<ov::ITensor> m_tensor;

public:
    VectorImpl(const std::shared_ptr<ov::ITensor>& tensor) : m_tensor(tensor) {}

    template <class T>
    operator const std::vector<T>&() {
        auto impl = std::dynamic_pointer_cast<VectorTensorImpl<T>>(m_tensor);
        OPENVINO_ASSERT(impl, "Cannot get vector. Type is incorrect!");
        return impl->get();
    }

    void set_shape(const ov::Shape& shape) override {
        m_tensor->set_shape(shape);
    }

    const element::Type& get_element_type() const override {
        return m_tensor->get_element_type();
    }

    const Shape& get_shape() const override {
        return m_tensor->get_shape();
    }

    size_t get_size() const override {
        return m_tensor->get_size();
    }

    size_t get_byte_size() const override {
        return m_tensor->get_byte_size();
    }

    Coordinate get_offsets() const override {
        return m_tensor->get_offsets();
    }

    Strides get_strides() const override {
        return m_tensor->get_strides();
    }

    void* data(const element::Type& type = {}) const override {
        return m_tensor->data();
    }

    template <typename T, typename datatype = typename std::decay<T>::type>
    T* data() const {
        return static_cast<T*>(data(element::from<datatype>()));
    }

    AnyMap get_properties() const override {
        return m_tensor->get_properties();
    }
};

}  // namespace template_plugin
}  // namespace ov

std::shared_ptr<ov::template_plugin::VectorTensor::VectorImpl> ov::template_plugin::VectorTensor::get_impl() const {
    auto impl = std::dynamic_pointer_cast<ov::template_plugin::VectorTensor::VectorImpl>(_impl);
    OPENVINO_ASSERT(impl);
    return impl;
}

ov::template_plugin::RemoteContext::RemoteContext() : m_name("TEMPLATE") {}

const std::string& ov::template_plugin::RemoteContext::get_device_name() const {
    return m_name;
}
const ov::AnyMap& ov::template_plugin::RemoteContext::get_property() const {
    return m_property;
}

std::shared_ptr<ov::ITensor> ov::template_plugin::RemoteContext::create_tensor(const ov::element::Type& type,
                                                                               const ov::Shape& shape,
                                                                               const ov::AnyMap& params) {
    std::shared_ptr<ov::ITensor> tensor;

    switch (type) {
    case ov::element::boolean:
        tensor =
            std::make_shared<VectorTensorImpl<ov::element_type_traits<ov::element::boolean>::value_type>>(type, shape);
        break;
    case ov::element::bf16:
        tensor =
            std::make_shared<VectorTensorImpl<ov::element_type_traits<ov::element::bf16>::value_type>>(type, shape);
        break;
    case ov::element::f16:
        tensor = std::make_shared<VectorTensorImpl<ov::element_type_traits<ov::element::f16>::value_type>>(type, shape);
        break;
    case ov::element::f32:
        tensor = std::make_shared<VectorTensorImpl<ov::element_type_traits<ov::element::f32>::value_type>>(type, shape);
        break;
    case ov::element::f64:
        tensor = std::make_shared<VectorTensorImpl<ov::element_type_traits<ov::element::f64>::value_type>>(type, shape);
        break;
    case ov::element::i8:
        tensor = std::make_shared<VectorTensorImpl<ov::element_type_traits<ov::element::i8>::value_type>>(type, shape);
        break;
    case ov::element::i16:
        tensor = std::make_shared<VectorTensorImpl<ov::element_type_traits<ov::element::i16>::value_type>>(type, shape);
        break;
    case ov::element::i32:
        tensor = std::make_shared<VectorTensorImpl<ov::element_type_traits<ov::element::i32>::value_type>>(type, shape);
        break;
    case ov::element::i64:
        tensor = std::make_shared<VectorTensorImpl<ov::element_type_traits<ov::element::i64>::value_type>>(type, shape);
        break;
    case ov::element::u8:
        tensor = std::make_shared<VectorTensorImpl<ov::element_type_traits<ov::element::u8>::value_type>>(type, shape);
        break;
    case ov::element::u16:
        tensor = std::make_shared<VectorTensorImpl<ov::element_type_traits<ov::element::u16>::value_type>>(type, shape);
        break;
    case ov::element::u32:
        tensor = std::make_shared<VectorTensorImpl<ov::element_type_traits<ov::element::u32>::value_type>>(type, shape);
        break;
    case ov::element::u64:
        tensor = std::make_shared<VectorTensorImpl<ov::element_type_traits<ov::element::u64>::value_type>>(type, shape);
        break;
    default:
        OPENVINO_UNREACHABLE("Cannot create remote tensor for unsupported type: ", type);
    }
    return std::make_shared<ov::template_plugin::VectorTensor::VectorImpl>(tensor);
}
