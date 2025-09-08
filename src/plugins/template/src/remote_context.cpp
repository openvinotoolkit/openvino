// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "remote_context.hpp"

#include <memory>

#include "openvino/core/any.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type_traits.hpp"
#include "openvino/runtime/itensor.hpp"
#include "remote_tensor.hpp"
#include "template/remote_tensor.hpp"

namespace ov {
namespace template_plugin {

// ! [vector_impl_t:implementation]
template <class T>
class VectorTensorImpl : public ov::IRemoteTensor {
    void update_strides() {
        if (m_element_type.bitwidth() < 8)
            return;
        auto& shape = get_shape();
        m_strides.clear();
        if (!shape.empty()) {
            m_strides.resize(shape.size());
            m_strides.back() = shape.back() == 0 ? 0 : m_element_type.size();
            std::copy(shape.rbegin(), shape.rend() - 1, m_strides.rbegin() + 1);
            std::partial_sum(m_strides.rbegin(), m_strides.rend(), m_strides.rbegin(), std::multiplies<size_t>());
        }
    }
    ov::element::Type m_element_type;
    ov::Shape m_shape;
    ov::Strides m_strides;
    std::vector<T> m_data;
    std::string m_dev_name;
    ov::AnyMap m_properties;

public:
    VectorTensorImpl(const ov::element::Type element_type, const ov::Shape& shape)
        : m_element_type{element_type},
          m_shape{shape},
          m_data(ov::shape_size(shape)),
          m_dev_name("TEMPLATE"),
          m_properties{{ov::device::full_name.name(), m_dev_name},
                       {"vector_data", m_data},
                       {"vector_data_ptr", static_cast<void*>(m_data.data())}} {
        update_strides();
    }

    const ov::element::Type& get_element_type() const override {
        return m_element_type;
    }

    const ov::Shape& get_shape() const override {
        return m_shape;
    }
    const ov::Strides& get_strides() const override {
        OPENVINO_ASSERT(m_element_type.bitwidth() >= 8,
                        "Could not get strides for types with bitwidths less then 8 bit. Tensor type: ",
                        m_element_type);
        return m_strides;
    }

    void set_shape(ov::Shape new_shape) override {
        auto old_byte_size = get_byte_size();
        OPENVINO_ASSERT(shape_size(new_shape) * get_element_type().size() <= old_byte_size,
                        "Could set new shape: ",
                        new_shape);
        m_shape = std::move(new_shape);
        update_strides();
    }

    const ov::AnyMap& get_properties() const override {
        return m_properties;
    }

    const std::string& get_device_name() const override {
        return m_dev_name;
    }
};
// ! [vector_impl_t:implementation]

// ! [remote_context:ctor]
RemoteContext::RemoteContext() : m_name("TEMPLATE") {}
// ! [remote_context:ctor]

// ! [remote_context:get_device_name]
const std::string& RemoteContext::get_device_name() const {
    return m_name;
}
// ! [remote_context:get_device_name]

// ! [remote_context:get_property]
const ov::AnyMap& RemoteContext::get_property() const {
    return m_property;
}
// ! [remote_context:get_property]

// ! [remote_context:create_tensor]
ov::SoPtr<ov::IRemoteTensor> RemoteContext::create_tensor(const ov::element::Type& type,
                                                          const ov::Shape& shape,
                                                          const ov::AnyMap& params) {
    std::shared_ptr<ov::IRemoteTensor> tensor;

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
        OPENVINO_THROW("Cannot create remote tensor for unsupported type: ", type);
    }
    return std::make_shared<VectorImpl>(tensor);
}
// ! [remote_context:create_tensor]

}  // namespace template_plugin
}  // namespace ov
