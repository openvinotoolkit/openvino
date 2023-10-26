// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/variable_value.hpp"

#include <memory>

#include "ngraph/runtime/host_tensor.hpp"
#include "openvino/core/deprecated.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/runtime/allocator.hpp"
#include "openvino/runtime/itensor.hpp"
#include "openvino/runtime/tensor.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START
namespace {

class TensorWrapper : public ngraph::runtime::HostTensor {
public:
    TensorWrapper(const ov::Tensor& tensor)
        : ngraph::runtime::HostTensor(tensor.get_element_type(), tensor.get_shape(), tensor.data()),
          tensor(tensor) {}

    ov::Tensor tensor;
};

/**
 * @brief Tensor what contains HostTensorPtr inside
 */
class HostTensorWrapper : public ov::ITensor {
public:
    ngraph::HostTensorPtr tensor;

    HostTensorWrapper(const ngraph::HostTensorPtr& tensor) : tensor{tensor}, m_type(tensor->get_element_type()) {
        const auto& p_shape = tensor->get_partial_shape();
        if (p_shape.is_static()) {
            m_shape = p_shape.to_shape();
        } else {
            OPENVINO_SUPPRESS_DEPRECATED_START
            m_shape = ov::util::make_dynamic_shape();
            OPENVINO_SUPPRESS_DEPRECATED_END
        }
        update_strides();
    }

    const ov::element::Type& get_element_type() const override {
        return m_type;
    }

    void set_shape(ov::Shape shape) override {
        tensor->set_shape(shape);
        m_shape = shape;
        update_strides();
    }

    const ov::Shape& get_shape() const override {
        return m_shape;
    }

    const ov::Strides& get_strides() const override {
        OPENVINO_ASSERT(get_element_type().bitwidth() >= 8,
                        "Could not get strides for types with bitwidths less then 8 bit. Tensor type: ",
                        get_element_type());
        return m_strides;
    }

    size_t get_size() const override {
        return ov::shape_size(m_shape);
    }

    size_t get_byte_size() const override {
        return get_size() * m_type.size();
    }

    void* data(const ov::element::Type& element_type) const override {
        return tensor->get_data_ptr();
    }

private:
    ov::element::Type m_type;
    ov::Shape m_shape;
    ov::Strides m_strides;

    void update_strides() {
        if (m_type.bitwidth() >= 8) {
            m_strides.clear();
            m_strides.resize(m_shape.size());
            auto size = m_strides.size();
            for (size_t i = 0; i < size; i++) {
                size_t value(m_type.size());
                size_t dim(m_shape[size - 1 - i]);
                if (i) {
                    value = m_strides[size - i] * dim;
                }
                m_strides[size - i - 1] = value;
            }
        }
    }
};
}  // namespace

ov::op::util::VariableValue::VariableValue() = default;

OPENVINO_SUPPRESS_DEPRECATED_START
ov::op::util::VariableValue::VariableValue(ngraph::HostTensorPtr value)
    : m_value(ov::Tensor{std::make_shared<HostTensorWrapper>(value), {}}) {}

ov::op::util::VariableValue::VariableValue(ngraph::HostTensorPtr value, bool reset)
    : m_reset(reset),
      m_value(ov::Tensor{std::make_shared<HostTensorWrapper>(value), {}}) {}

ngraph::HostTensorPtr ov::op::util::VariableValue::get_value() const {
    if (auto wrapper = std::dynamic_pointer_cast<HostTensorWrapper>(m_value._impl))
        return wrapper->tensor;
    return std::make_shared<TensorWrapper>(m_value);
}

void ov::op::util::VariableValue::set_value(const ngraph::HostTensorPtr& value) {
    m_value = ov::Tensor{std::make_shared<HostTensorWrapper>(value), {}};
}
OPENVINO_SUPPRESS_DEPRECATED_END

void ov::op::util::VariableValue::set_reset(bool reset) {
    m_reset = reset;
}

bool ov::op::util::VariableValue::get_reset() const {
    return m_reset;
}

ov::op::util::VariableValue::VariableValue(const ov::Tensor& value) : m_value(value) {}

ov::op::util::VariableValue::VariableValue(const ov::Tensor& value, bool reset) : m_reset(reset), m_value(value) {}

const ov::Tensor& ov::op::util::VariableValue::get_state() const {
    return m_value;
}

void ov::op::util::VariableValue::set_state(const ov::Tensor& value) {
    m_value = value;
}
