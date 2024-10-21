// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/remote_tensor.hpp"

#include "intel_npu/config/common.hpp"
#include "openvino/core/type/element_iterator.hpp"
#include "openvino/runtime/intel_npu/remote_properties.hpp"

namespace intel_npu {

RemoteTensor::RemoteTensor(std::shared_ptr<ov::IRemoteContext> context,
                           const ov::element::Type& element_type,
                           const ov::Shape& shape)
    : _context(std::move(context)),
      _element_type(element_type),
      _shape(shape),
      _capacity(shape) {
    OPENVINO_ASSERT(shape_size(_shape) != 0);
    OPENVINO_ASSERT(_element_type != ov::element::undefined && _element_type.is_static());
}

RemoteTensor::~RemoteTensor() = default;

const ov::element::Type& RemoteTensor::get_element_type() const {
    return _element_type;
}

const ov::Shape& RemoteTensor::get_shape() const {
    return _shape;
}

const ov::Strides& RemoteTensor::get_strides() const {
    return _strides;
}

const ov::AnyMap& RemoteTensor::get_properties() const {
    return _properties;
}

void RemoteTensor::set_shape(ov::Shape new_shape) {
    _shape = std::move(new_shape);

    if (ov::shape_size(_shape) > ov::shape_size(_capacity)) {
        if (!deallocate()) {
            OPENVINO_THROW("Cannot deallocate tensor while an attempt to enlarge tensor area in set_shape.");
        }

        const auto byte_size = ov::element::get_memory_size(_element_type, shape_size(_shape));
        allocate(byte_size);
    } else {
        _strides.clear();
        update_strides();
    }
}

void RemoteTensor::update_strides() {
    if (_element_type.bitwidth() < 8) {
        return;
    }

    auto& shape = get_shape();
    if (_strides.empty() && !shape.empty()) {
        _strides.resize(shape.size());
        _strides.back() = shape.back() == 0 ? 0 : _element_type.size();
        std::transform(shape.crbegin(),
                       shape.crend() - 1,
                       _strides.rbegin(),
                       _strides.rbegin() + 1,
                       std::multiplies<size_t>());
    }
}

const std::string& RemoteTensor::get_device_name() const {
    return _context->get_device_name();
}

std::shared_ptr<ov::IRemoteContext> RemoteTensor::get_context() const {
    return _context;
}

}  // namespace intel_npu
