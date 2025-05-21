// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "common_utils/dispatch_utils.hpp"
#include "common_utils/jit_term.hpp"
#include "common_utils/jitter.hpp"
#include "intel_gpu/runtime/layout.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_gpu::ocl {

using cldnn::format;
using cldnn::layout;

template <typename T>
inline std::string get_ocl_type_name() {
    throw std::runtime_error("Implement me");
}
template <>
inline std::string get_ocl_type_name<int8_t>() {
    return "char";
}
template <>
inline std::string get_ocl_type_name<uint8_t>() {
    return "uchar";
}
template <>
inline std::string get_ocl_type_name<int16_t>() {
    return "short";
}
template <>
inline std::string get_ocl_type_name<uint16_t>() {
    return "ushort";
}
template <>
inline std::string get_ocl_type_name<int32_t>() {
    return "int";
}
template <>
inline std::string get_ocl_type_name<uint32_t>() {
    return "uint";
}
template <>
inline std::string get_ocl_type_name<int64_t>() {
    return "long";
}
template <>
inline std::string get_ocl_type_name<uint64_t>() {
    return "ulong";
}
template <>
inline std::string get_ocl_type_name<float>() {
    return "float";
}
template <>
inline std::string get_ocl_type_name<ov::float16>() {
    return "half";
}
template <>
inline std::string get_ocl_type_name<double>() {
    return "double";
}

std::string to_ocl_type(ov::element::Type_t et);

class LayoutJitter {
public:
    // definition of tensor element accessors in the following order:
    // data tensor: b, f, u, v, w, z, y, x
    // weights tensor: g, ofm, ifm, z, y, x
    std::vector<JitTerm> m_dims;
    std::vector<JitTerm> m_strides;
    std::vector<JitTerm> m_pad_lower;
    std::vector<JitTerm> m_pad_upper;
    JitTerm m_offset;

    LayoutJitter(const layout& l, size_t shape_info_offset) {
        OPENVINO_ASSERT(!format::is_weights_format(l.format));
        make_definitions(l, shape_info_offset);
    }

    std::map<ChannelName, size_t> channels_map;

    [[nodiscard]] std::string dim(ChannelName channel) const {
        return m_dims[channels_map.at(channel)].str();
    }

    [[nodiscard]] std::string pad_l(ChannelName channel) const {
        return m_pad_lower[channels_map.at(channel)].str();
    }

    [[nodiscard]] std::string pad_u(ChannelName channel) const {
        return m_pad_upper[channels_map.at(channel)].str();
    }

    [[nodiscard]] std::string stride(ChannelName channel) const {
        return m_strides[channels_map.at(channel)].str();
    }

    [[nodiscard]] std::string offset() const {
        return m_offset.str();
    }

private:
    void make_definitions(const layout& l, size_t shape_info_offset);
};

inline JitTerm make_type(ov::element::Type dt, size_t vec_size = 1) {
    if (vec_size > 1) {
        return concat(dt, vec_size);
    }
    return JitTerm{to_ocl_type(dt)};
}

inline JitTerm make_type(const JitTerm& scalar_type, size_t vec_size = 1) {
    const JitTerm make_vector_type_f{"MAKE_VECTOR_TYPE"};
    if (vec_size > 1) {
        return make_vector_type_f(scalar_type, vec_size);
    }
    return scalar_type;
}

inline JitTerm make_const_global_ptr(ov::element::Type_t type, const JitTerm& name = {}) {
    return concat("const __global ", to_ocl_type(type), "* ", name);
}

inline JitTerm make_const_global_ptr(ov::element::Type_t type, size_t vec_size, const JitTerm& name = {}) {
    if (vec_size == 1) {
        return make_const_global_ptr(type, name);
    }
    return concat("const __global ", to_ocl_type(type), vec_size, "* ", name);
}

inline JitTerm global_ptr_cast(ov::element::Type_t type, size_t vec_size, const JitTerm& ptr) {
    return concat("(", make_const_global_ptr(type, vec_size), ")")(ptr);
}

inline JitTerm cast_to_type(const JitTerm& var, ov::element::Type dt, size_t vec_size = 1) {
    return concat("as_", make_type(dt, vec_size)(var));
}

inline JitTerm convert_to_type(const JitTerm& var, ov::element::Type dt, size_t vec_size = 1) {
    return concat("convert_", make_type(dt, vec_size)(var));
}

inline JitTerm make_block_read(ov::element::Type_t type, size_t vec_size, const JitTerm& ptr) {
    const JitTerm read_f_base("_sub_group_block_read");

    if (ov::element::Type(type).size() == ov::element::u32.size()) {
        const JitTerm read_f = concat(read_f_base, vec_size);
        return cast_to_type(read_f(global_ptr_cast(ov::element::u32, vec_size, ptr)), type, vec_size);
    }

    if (ov::element::Type(type).size() == ov::element::u16.size()) {
        const JitTerm read_f = concat(read_f_base, "_us", vec_size);
        return cast_to_type(read_f(global_ptr_cast(ov::element::u16, vec_size, ptr)), type, vec_size);
    }

    if (ov::element::Type(type).size() == ov::element::u8.size()) {
        const JitTerm read_f = concat(read_f_base, "_uc", vec_size);
        return cast_to_type(read_f(global_ptr_cast(ov::element::u8, vec_size, ptr)), type, vec_size);
    }

    OPENVINO_THROW("[GPU] Unexpected element size for block read: ", type);
}

inline JitTerm broadcast(const JitTerm& var, ov::element::Type dt, size_t vec_size) {
    return concat("(", make_type(dt, vec_size), ")")(var);
}

inline JitTerm declare_var(const JitTerm& type, const JitTerm& name, const JitTerm& value = {}) {
    if (value.str().empty()) {
        return concat(type, " ", name);
    }
    return concat(type, " ", name, " = ", value);
}

JitConstants make_layout_jit_constants(const std::string& name, const cldnn::layout& value, size_t shape_info_offset);
JitConstants make_type_jit_constants(const std::string& name, const ov::element::Type& value);
JitConstants make_indexing_jit_functions(const std::string& name, const layout& l);
JitConstants make_int4_packed_type_jit_constant(const std::string& macro_name, ov::element::Type type, size_t pack_size);
}  // namespace ov::intel_gpu::ocl

namespace ov::intel_gpu {
template <>
inline std::string to_code_string(const ov::element::Type& val) {
    return ocl::to_ocl_type(val);
}
template <>
inline std::string to_code_string(const ov::element::Type_t& val) {
    return ocl::to_ocl_type(val);
}

}  // namespace ov::intel_gpu
