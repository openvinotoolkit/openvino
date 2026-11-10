// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/itensor.hpp"

#include <memory>

#include "compare.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/memory_util.hpp"
#include "openvino/runtime/iremote_tensor.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/properties.hpp"

namespace ov {

ITensor::~ITensor() = default;

size_t ITensor::get_size() const {
    return shape_size(get_shape());
}

size_t ITensor::get_byte_size() const {
    return util::get_memory_size(get_element_type(), get_size());
}

bool ITensor::is_continuous() const {
    const auto& type = get_element_type();
    const auto& shape = get_shape();
    if (shape_size(shape) == 0 || !util::can_compute_byte_strides(type, shape)) {
        return true;
    }

    const auto& strides = get_strides();
    auto stride = strides.rbegin();
    const auto default_strides = util::get_dense_byte_strides(type, shape);
    auto default_stride = default_strides.rbegin();

    for (; stride != strides.rend(); ++stride, ++default_stride) {
        if (*stride != *default_stride) {
            break;
        }
    }

    const auto default_last = default_strides.rend();
    // It assumed that `default_strides' and `strides' have the same size, thus `default_stride' iterator is valid
    // coverity[deref_iterator:SUPPRESS]
    return (default_stride == default_last) || (*default_stride < *stride && (shape[0] == 1) &&
                                                std::all_of(default_stride, default_last, cmp::Equal(*default_stride)));
}

void ITensor::copy_to(const std::shared_ptr<ov::ITensor>& dst) const {
    const auto& is_scalar = [](const ov::Shape& shape) {
        return shape.empty() || (shape.size() == 1 && shape[0] == 1);
    };
    OPENVINO_ASSERT(dst, "Destination tensor was not initialized.");
    OPENVINO_ASSERT(!dynamic_cast<const ov::IRemoteTensor*>(this),
                    "Default copy to doesn't support copy from remote tensor.");

    const auto& type = get_element_type();
    const auto& shape = get_shape();

    OPENVINO_ASSERT(dst->get_element_type() == type,
                    "Tensor element types are not equal. (src: ",
                    type,
                    " != dst: ",
                    dst->get_element_type(),
                    ")");

    if (shape != dst->get_shape()) {
        dst->set_shape(shape);
    }

    if (get_size() == 0) {
        return;
    }

    if (auto remote_tensor_dst = std::dynamic_pointer_cast<ov::IRemoteTensor>(dst)) {
        remote_tensor_dst->copy_from(shared_from_this());
        return;
    }

    if (shape_size(shape) == 0) {
        return;
    }

    auto* src_data = static_cast<const uint8_t*>(data());
    auto* dst_data = static_cast<uint8_t*>(dst->data());
    ov::Strides src_strides{get_byte_size()};
    ov::Strides dst_strides{dst->get_byte_size()};
    ov::Shape cur_pos{0};
    ov::Shape max_pos{1};

    if (!util::can_compute_byte_strides(type, shape) || (get_strides() == dst->get_strides() && is_continuous()) ||
        (is_scalar(shape) && is_scalar(dst->get_shape()))) {
        // byte strides are not representable for this LP type/shape or both tensors have default strides
        // Strides and positions already initialized
    } else {
        // Tensors have default strides
        const auto shape_rank = shape.size();
        const auto default_strides = util::get_dense_byte_strides(type, shape);

        // for sub-byte types the last dimension's element count does not match its real byte
        // span (default_strides.back() is an unaddressable placeholder) - use its packed byte
        // count instead, so dimension counts below always line up with real byte strides
        const auto byte_shape = util::get_byte_memory_shape(type, shape);

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
                        dim = byte_shape[strides_size];
                    }
                    src_str[strides_size] = strides;
                    dst_str[strides_size] = strides;
                    max_pos[strides_size] = dim;
                    cur_pos[strides_size] = 0;
                }
            }
            src_str[inverted_idx] = src_strides[inverted_idx];
            dst_str[inverted_idx] = dst_strides[inverted_idx];
            max_pos[inverted_idx] = byte_shape[inverted_idx];
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
