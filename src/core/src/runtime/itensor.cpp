// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/itensor.hpp"

#include <memory>

#include "openvino/core/except.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/core/type/element_iterator.hpp"
#include "openvino/runtime/allocator.hpp"
#include "openvino/runtime/iremote_tensor.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/properties.hpp"

namespace ov {

namespace {
Strides default_byte_strides(const Shape& shape, const element::Type& et) {
    auto strides = Strides(shape.size());
    if (!strides.empty()) {
        strides.back() = et.size();
        std::transform(shape.crbegin(),
                       shape.crend() - 1,
                       strides.rbegin(),
                       strides.rbegin() + 1,
                       std::multiplies<size_t>());
    }
    return strides;
}
}  // namespace

ITensor::~ITensor() = default;

size_t ITensor::get_size() const {
    return shape_size(get_shape());
}

size_t ITensor::get_byte_size() const {
    return element::get_memory_size(get_element_type(), get_size());
}

bool ITensor::is_continuous() const {
    if ((get_element_type().bitwidth() < 8) || get_size() == 0) {
        // OpenVINO doesn't support strides for lp types
        return true;
    }
    return default_byte_strides(get_shape(), get_element_type()) == get_strides();
}

void ITensor::copy_to(const std::shared_ptr<ov::ITensor>& dst) const {
    const auto& is_scalar = [](const ov::Shape& shape) {
        return shape.empty() || (shape.size() == 1 && shape[0] == 1);
    };
    OPENVINO_ASSERT(dst, "Destination tensor was not initialized.");
    OPENVINO_ASSERT(!dynamic_cast<const ov::IRemoteTensor*>(this),
                    "Default copy to doesn't support copy from remote tensor.");
    OPENVINO_ASSERT(dst->get_element_type() == get_element_type(),
                    "Tensor element types are not equal. (src: ",
                    get_element_type(),
                    " != dst: ",
                    dst->get_element_type(),
                    ")");

    const auto& shape = get_shape();
    if (shape != dst->get_shape()) {
        dst->set_shape(shape);
    }

    if (std::dynamic_pointer_cast<ov::IRemoteTensor>(dst)) {
        auto remote_tensor_dst = std::dynamic_pointer_cast<ov::IRemoteTensor>(dst);
        remote_tensor_dst->copy_from(shared_from_this());
        return;
    }

    auto* src_data = static_cast<const uint8_t*>(data());
    auto* dst_data = static_cast<uint8_t*>(dst->data());
    ov::Strides src_strides{get_byte_size()};
    ov::Strides dst_strides{dst->get_byte_size()};
    ov::Shape cur_pos{0};
    ov::Shape max_pos{1};

    if (get_element_type().bitwidth() < 8 || (get_strides() == dst->get_strides() && is_continuous()) ||
        (is_scalar(shape) && is_scalar(dst->get_shape()))) {
        // OpenVINO doesn't support strides for LP types
        // or both tensors have default strides
        // Strides and positions already initialized
    } else {
        // Tensors have default strides
        const auto& type = get_element_type();
        const auto shape_rank = shape.size();
        const auto default_strides = default_byte_strides(shape, type);

        src_strides = get_strides();
        dst_strides = dst->get_strides();

        ov::Strides src_str, dst_str;

        // Calculate src and dst shapes
        bool found_step = false;
        for (size_t inverted_idx = shape_rank - 1; inverted_idx < shape_rank; --inverted_idx) {
            if (!found_step) {
                if (default_strides[inverted_idx] == src_strides[inverted_idx] &&
                    src_strides[inverted_idx] == dst_strides[inverted_idx]) {
                    continue;
                } else {
                    found_step = true;
                    size_t strides_size = inverted_idx + 1;
                    // Set right size
                    src_str.resize(strides_size + 1);
                    dst_str.resize(strides_size + 1);
                    max_pos.resize(strides_size + 1);
                    cur_pos.resize(strides_size + 1);
                    // In case of default continuous strides we can copy several elements
                    // In other case only one element
                    size_t dim = 1;
                    size_t strides = type.size();

                    if (strides_size < default_strides.size()) {
                        strides = default_strides[strides_size];
                        dim = shape[strides_size];
                    }
                    src_str[strides_size] = strides;
                    dst_str[strides_size] = strides;
                    max_pos[strides_size] = dim;
                    cur_pos[strides_size] = 0;
                }
            }
            src_str[inverted_idx] = src_strides[inverted_idx];
            dst_str[inverted_idx] = dst_strides[inverted_idx];
            max_pos[inverted_idx] = shape[inverted_idx];
            cur_pos[inverted_idx] = 0;
        }
        src_strides = std::move(src_str);
        dst_strides = std::move(dst_str);
    }

    const auto update_index = [](const ov::Shape& pos, const ov::Strides& strides) {
        return std::inner_product(pos.begin(), pos.end(), strides.begin(), static_cast<size_t>(0));
    };

    using copy_function_def = std::function<void(const uint8_t*, uint8_t*, size_t)>;
    copy_function_def memcpy_based_copy = [](const uint8_t* src_data, uint8_t* dst_data, size_t bytes_size) {
        memcpy(dst_data, src_data, bytes_size);
    };
    copy_function_def strings_copy = [](const uint8_t* src_data, uint8_t* dst_data, size_t bytes_size) {
        // in case string tensors, it needs to copy of new values for std::string objects
        // memcpy is not suitable
        auto dst_string = reinterpret_cast<std::string*>(dst_data);
        auto src_string = reinterpret_cast<const std::string*>(src_data);
        size_t num_elements_stride = bytes_size / element::string.size();
        std::copy_n(src_string, num_elements_stride, dst_string);
    };
    copy_function_def copy_function = (get_element_type() == element::string) ? strings_copy : memcpy_based_copy;

    bool finish = false;
    for (size_t dst_idx = 0, src_idx = 0; !finish;) {
        copy_function(src_data + src_idx, dst_data + dst_idx, src_strides[src_strides.size() - 1]);

        // update indexes
        for (size_t i = 0; i < cur_pos.size(); i++) {
            size_t inverted_idx = cur_pos.size() - i - 1;
            cur_pos[inverted_idx]++;
            if (cur_pos[inverted_idx] != max_pos[inverted_idx]) {
                break;
            }
            if (inverted_idx)
                cur_pos[inverted_idx] = 0;
            else
                finish = true;
        }
        src_idx = update_index(cur_pos, src_strides);
        dst_idx = update_index(cur_pos, dst_strides);
    }
}

}  // namespace ov
