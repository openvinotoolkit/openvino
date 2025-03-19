// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/remote_tensor.hpp"
#include "intel_gpu/runtime/layout.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/memory_caps.hpp"

#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/op/util/op_types.hpp"

#include <algorithm>
#include <memory>

namespace {

template <typename src_t, typename dst_t>
void convert_and_copy_no_pad(const src_t* src, dst_t* dst, size_t size) {
    OPENVINO_ASSERT(src && dst, "[GPU] Src or Dst ptr is null");
    for (size_t i = 0; i < size; i++)
        dst[i] = static_cast<dst_t>(src[i]);
}

template <typename src_t, typename dst_t>
void convert_and_copy_padded_source(const src_t* src, dst_t* dst, cldnn::layout layout) {
    cldnn::tensor size = layout.get_tensor();
    for (int64_t b = 0; b < size.batch[0]; b++) {
        for (int64_t f = 0; f < size.feature[0]; f++) {
            for (int64_t w = 0; w < size.spatial[3]; w++) {
                for (int64_t z = 0; z < size.spatial[2]; z++) {
                    for (int64_t y = 0; y < size.spatial[1]; y++) {
                        for (int64_t x = 0; x < size.spatial[0]; x++) {
                            *dst++ = static_cast<dst_t>(src[layout.get_linear_offset(cldnn::tensor(b, f, x, y, z, w))]);
                        }
                    }
                }
            }
        }
    }
}

void convert_and_copy(const void* src_ptr, ov::element::Type src_et, void* dst_ptr, ov::element::Type dst_et, size_t size, cldnn::layout layout) {
    if (size == 0)
        return;

    if (src_et == dst_et && !layout.data_padding) {
        std::memcpy(dst_ptr, src_ptr, size * src_et.size());
        return;
    }

    #define CASE(s_et, d_et, s_type, d_type)                                                                                       \
        if (src_et == s_et && dst_et == d_et) {                                                                                    \
            if (static_cast<bool>(layout.data_padding)) {                                                                          \
                return convert_and_copy_padded_source(static_cast<const s_type*>(src_ptr), static_cast<d_type*>(dst_ptr), layout); \
            } else {                                                                                                               \
                return convert_and_copy_no_pad(static_cast<const s_type*>(src_ptr), static_cast<d_type*>(dst_ptr), size);          \
            }                                                                                                                      \
        }

    // For unsupported inputs
    CASE(ov::element::f64, ov::element::f32, double, float);
    CASE(ov::element::i16, ov::element::f32, int16_t, float);
    CASE(ov::element::u16, ov::element::f32, uint16_t, float);
    CASE(ov::element::u64, ov::element::i32, uint64_t, int32_t);
    CASE(ov::element::i64, ov::element::i32, int64_t, int32_t);
    CASE(ov::element::u32, ov::element::i32, uint32_t, int32_t);

    // For unsupported outputs
    CASE(ov::element::f32, ov::element::f64, float, double);
    CASE(ov::element::i32, ov::element::i64, int32_t, int64_t);
    CASE(ov::element::i32, ov::element::u64, int32_t, uint64_t);
    CASE(ov::element::i32, ov::element::u32, int32_t, uint32_t);
    CASE(ov::element::f32, ov::element::i16, float, int16_t);
    CASE(ov::element::f32, ov::element::u16, float, uint16_t);

    // TODO: Need instances below?
    CASE(ov::element::u32, ov::element::i64, uint32_t, int64_t);
    CASE(ov::element::u32, ov::element::u64, uint32_t, uint64_t);

    // For state conversions
    CASE(ov::element::f32, ov::element::f32, float, float);
    CASE(ov::element::f32, ov::element::f16, float, ov::float16);
    CASE(ov::element::f16, ov::element::f32, ov::float16, float);
    CASE(ov::element::f16, ov::element::f16, ov::float16, ov::float16);
    CASE(ov::element::bf16, ov::element::f32, ov::bfloat16, float);
    CASE(ov::element::bf16, ov::element::f16, ov::bfloat16, ov::float16);
    CASE(ov::element::boolean, ov::element::u8, bool, uint8_t);

    OPENVINO_THROW("[GPU] Unsupported element types combination for copy: ", src_et, " -> ", dst_et);
}

}  // namespace

namespace ov::intel_gpu {

bool is_supported(ov::element::Type_t et) {
    switch (et) {
        case ov::element::Type_t::dynamic: return true;
        case ov::element::Type_t::boolean: return true; // converted to u8
        case ov::element::Type_t::bf16: return false;
        case ov::element::Type_t::f16: return true;
        case ov::element::Type_t::f32: return true;
        case ov::element::Type_t::f64: return true; // converted to inference precision
        case ov::element::Type_t::i4: return true;
        case ov::element::Type_t::i8: return true;
        case ov::element::Type_t::i16: return false;
        case ov::element::Type_t::i32: return true;
        case ov::element::Type_t::i64: return true; // converted to i32
        case ov::element::Type_t::u1: return true;
        case ov::element::Type_t::u2: return false;
        case ov::element::Type_t::u3: return false;
        case ov::element::Type_t::u4: return true;
        case ov::element::Type_t::u6: return true;
        case ov::element::Type_t::u8: return true;
        case ov::element::Type_t::u16: return true; // converted to i32
        case ov::element::Type_t::u32: return true; // converted to i32
        case ov::element::Type_t::u64: return true; // converted to i32
        case ov::element::Type_t::nf4: return false;
        case ov::element::Type_t::f8e4m3: return false;
        case ov::element::Type_t::f8e5m2: return false;
        case ov::element::Type_t::string: return false;
        default: return false;
    }

    return false;
}

bool data_types_are_supported(const ov::Node* node) {
    for (size_t i = 0; i < node->get_input_size(); i++) {
        if (!is_supported(node->get_input_element_type(i)))
            return false;
    }

    for (size_t i = 0; i < node->get_output_size(); i++) {
        if (!is_supported(node->get_output_element_type(i)))
            return false;
    }

    return true;
}

void convert_and_copy(const ov::ITensor* src, cldnn::memory::ptr dst, cldnn::stream& stream, const cldnn::layout& src_layout) {
    const bool blocking = true;
    auto src_et = src->get_element_type();
    auto dst_et = dst->get_layout().data_type;

    if (dst_et == src_et) {
        if (auto remote = dynamic_cast<const ov::intel_gpu::RemoteTensorImpl*>(src)) {
            auto mem = remote->get_original_memory();
            dst->copy_from(stream, *mem, blocking);
        } else {
            dst->copy_from(stream, src->data(), blocking);
            return;
        }
    }

    size_t size = ov::shape_size(src->get_shape());
    ov::Tensor tmp_tensor(dst_et, src->get_shape());
    ::convert_and_copy(src->data(), src_et, tmp_tensor.data(), dst_et, size, src_layout);
    dst->copy_from(stream, tmp_tensor.data(), blocking);
}

void convert_and_copy(const cldnn::memory::ptr src, ov::ITensor const* dst, const cldnn::stream& stream) {
    auto src_et = src->get_layout().data_type;
    auto dst_et = dst->get_element_type();

    size_t size = ov::shape_size(dst->get_shape());

    cldnn::mem_lock<uint8_t, cldnn::mem_lock_type::read> src_lock(src, stream);
    std::unique_ptr<cldnn::mem_lock<uint8_t>> dst_lock = nullptr;

    const void* src_ptr = src_lock.data();
    void* dst_ptr = nullptr;

    if (auto remote = dynamic_cast<const ov::intel_gpu::RemoteTensorImpl*>(dst)) {
        auto mem = remote->get_original_memory();
        dst_lock.reset(new cldnn::mem_lock<uint8_t>(mem, stream));
        dst_ptr = dst_lock->data();
    } else {
        dst_ptr = dst->data();
    }

    return ::convert_and_copy(src_ptr, src_et, dst_ptr, dst_et, size, src->get_layout());
}

void convert_and_copy(const cldnn::memory::ptr src, cldnn::memory::ptr dst, cldnn::stream& stream) {
    const bool blocking = true;
    auto src_et = src->get_layout().data_type;
    auto dst_et = dst->get_layout().data_type;

    cldnn::mem_lock<uint8_t, cldnn::mem_lock_type::read> src_lock(src, stream);
    const void* src_ptr = src_lock.data();

    size_t size = ov::shape_size(src->get_layout().get_shape());
    ov::Tensor tmp_tensor(dst_et, src->get_layout().get_shape());
    ::convert_and_copy(src_ptr, src_et, tmp_tensor.data(), dst_et, size, src->get_layout());
    dst->copy_from(stream, tmp_tensor.data(), blocking);
}

void convert_and_copy(const ov::ITensor* src, ov::ITensor* dst, const cldnn::stream& stream) {
    auto src_et = src->get_element_type();
    auto dst_et = dst->get_element_type();

    size_t size = ov::shape_size(dst->get_shape());

    const void* src_ptr = nullptr;
    void* dst_ptr = nullptr;

    std::unique_ptr<cldnn::mem_lock<uint8_t, cldnn::mem_lock_type::read>> src_lock = nullptr;
    std::unique_ptr<cldnn::mem_lock<uint8_t>> dst_lock = nullptr;
    ov::Tensor tmp_tensor;

    if (auto remote = dynamic_cast<const ov::intel_gpu::RemoteTensorImpl*>(src)) {
        auto mem = remote->get_original_memory();
        src_lock.reset(new cldnn::mem_lock<uint8_t, cldnn::mem_lock_type::read>(mem, stream));
        src_ptr = src_lock->data();
    } else if (dynamic_cast<const ov::IRemoteTensor*>(src)) {
        tmp_tensor = ov::Tensor(src_et, src->get_shape());
        src->copy_to(get_tensor_impl(tmp_tensor)._ptr);
        src_ptr = tmp_tensor.data();
    } else {
        src_ptr = src->data();
    }

    if (auto remote = dynamic_cast<const ov::intel_gpu::RemoteTensorImpl*>(dst)) {
        auto mem = remote->get_original_memory();
        dst_lock.reset(new cldnn::mem_lock<uint8_t>(mem, stream));
        dst_ptr = dst_lock->data();
    } else if (auto remote = dynamic_cast<ov::IRemoteTensor*>(dst)) {
        tmp_tensor = ov::Tensor(dst_et, src->get_shape());
        ::convert_and_copy(src_ptr,
                           src_et,
                           tmp_tensor.data(),
                           dst_et,
                           size,
                           cldnn::layout({}, ov::element::dynamic, cldnn::format::bfyx, cldnn::padding()));
        remote->copy_from(get_tensor_impl(tmp_tensor)._ptr);
        return;
    } else {
        dst_ptr = dst->data();
    }

    return ::convert_and_copy(src_ptr,
                              src_et,
                              dst_ptr,
                              dst_et,
                              size,
                              cldnn::layout({}, ov::element::dynamic, cldnn::format::bfyx, cldnn::padding()));
}

std::vector<cldnn::optional_data_type> get_output_data_types(const ov::Node* op, PrecisionMap precision_map) {
    std::vector<cldnn::optional_data_type> output_data_types;
    for (size_t i = 0; i < op->get_output_size(); i++) {
        auto type = op->get_output_element_type(i);
        if (precision_map.find(type) != precision_map.end())
            type = precision_map.at(type);
        output_data_types.push_back(cldnn::element_type_to_data_type(type));
    }
    return output_data_types;
}

}  // namespace ov::intel_gpu
