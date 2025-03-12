// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <sstream>
#include <string>
#include <utility>

#include "common_utils/dispatch_utils.hpp"
#include "intel_gpu/primitives/activation.hpp"
#include "intel_gpu/runtime/layout.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_gpu {

using cldnn::layout;
using cldnn::format;
using cldnn::activation_func;

class CodeBuilder {
    std::ostringstream oss;
    std::string code;
    std::vector<std::string> defined_macroses;

    CodeBuilder& register_macro(const std::string& name) {
        assert(std::count(defined_macroses.begin(), defined_macroses.end(), name) == 0);
        defined_macroses.push_back(name);
        return *this;
    }

    CodeBuilder& unregister_macro(const std::string& name) {
        assert(std::count(defined_macroses.begin(), defined_macroses.end(), name) != 0);
        defined_macroses.erase(std::remove_if(defined_macroses.begin(),
                                              defined_macroses.end(),
                                              [&](const std::string& v) {
                                                  return v == name;
                                              }),
                               defined_macroses.end());
        return *this;
    }

public:
    CodeBuilder& set_code(const std::string& c) {
        assert(code.empty());
        code = c;
        return *this;
    }

    CodeBuilder& add_line(const std::string& line) {
        oss << line << "\n";
        return *this;
    }

    CodeBuilder& decoration_macro(const std::string& name,
                                  const std::string& prefix,
                                  const std::string& postfix,
                                  const std::string& name_prefix = std::string()) {
        oss << "#define " << name << "(name) " << prefix << " " + name_prefix + "_##" + "name" << (postfix.empty() ? "" : "##_") << postfix << '\n';
        return register_macro(name);
    }

    CodeBuilder& value_macro(const std::string& name, const std::string& value) {
        oss << "#define " << name << " " << value << '\n';
        return register_macro(name.substr(0, name.find('(')));
    }

    CodeBuilder& undef_macro(const std::string& name) {
        oss << "#undef " << name.substr(0, name.find('(')) << '\n';
        return unregister_macro(name.substr(0, name.find('(')));
    }

    std::string str() {
        oss << '\n';
        return oss.str();
    }
};

struct JitConstant {
    std::string name;
    std::string value;
    JitConstant(std::string n, std::string v) : name(std::move(n)), value(std::move(v)) {}
};

template <typename T>
std::string to_code_string(T val) {
    std::stringstream ss;
    ss.imbue(std::locale("C"));
    ss << val;
    return ss.str();
}

template <typename T>
JitConstant make_jit_constant(const std::string& name, T value) {
    return JitConstant(name, to_code_string(value));
}

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
inline std::string get_ocl_type_name<double>() {
    return "double";
}

std::string to_ocl_type(ov::element::Type_t et);

struct JitConstants : public std::vector<JitConstant> {
    void add(const JitConstant& constant) {
        push_back(constant);
    }

    void add(JitConstant&& constant) {
        push_back(std::move(constant));
    }

    template <typename... Args>
    void make(Args... args) {
        add(make_jit_constant(args...));
    }

    void add(const std::vector<JitConstant>& constants) {
        insert(end(), constants.begin(), constants.end());
    }

    void merge(const JitConstants& jit) {
        add(jit);
    }

    void remove(const std::string& name) {
        erase(std::remove_if(begin(),
                             end(),
                             [=](const JitConstant& x) -> bool {
                                 return x.name == name;
                             }),
              end());
    }

    JitConstants(std::initializer_list<JitConstant> values) : std::vector<JitConstant>(values) {}
    JitConstants() = default;
};

class JitTerm {
public:
    JitTerm() = default;
    explicit JitTerm(std::string text) : text(std::move(text)) {}

    [[nodiscard]] const std::string& str() const {
        return text;
    }

    [[nodiscard]] JitTerm gt(const JitTerm& rhs) const {
        JitTerm jit_term{"(" + text + ">" + rhs.str() + ")"};
        return jit_term;
    }

    [[nodiscard]] JitTerm ge(const JitTerm& rhs) const {
        JitTerm jit_term{"(" + text + ">=" + rhs.str() + ")"};
        return jit_term;
    }

    [[nodiscard]] JitTerm le(const JitTerm& rhs) const {
        JitTerm jit_term{"(" + text + "<=" + rhs.str() + ")"};
        return jit_term;
    }

    [[nodiscard]] JitTerm eq(const JitTerm& rhs) const {
        JitTerm jit_term{"(" + text + "==" + rhs.str() + ")"};
        return jit_term;
    }

    template <typename... Args>
    JitTerm operator()(Args... args) const {
        return JitTerm{text + "(" + concat(std::forward<Args>(args)..., ",").str() + ")"};
    }

    JitTerm operator[](const JitTerm& idx) const {
        JitTerm jit_term{text + "[" + idx.str() + "]"};
        return jit_term;
    }
    JitTerm operator[](size_t idx) const {
        JitTerm jit_term{text + "[" + to_code_string(idx) + "]"};
        return jit_term;
    }

private:
    template <typename... Args>
    JitTerm concat(const JitTerm& arg, Args... args, std::string separator) const {
        return concat(arg, concat(std::forward<Args>(args)..., separator), separator);
    }

    [[nodiscard]] static JitTerm concat(const JitTerm& t1, const JitTerm& t2, const std::string& separator) {
        return JitTerm{t1.str() + separator + t2.str()};
    }

    [[nodiscard]] static JitTerm concat(const JitTerm& t1, const std::string& separator) {
        return JitTerm{t1.str()};
    }

    [[nodiscard]] static JitTerm concat(const std::string& separator) {
        return JitTerm{""};
    }

    std::string text;
};

inline bool is_number(const JitTerm& s) {
    return !s.str().empty() && std::all_of(s.str().begin(), s.str().end(), ::isdigit);
}
template <typename T>
inline T as_number(const JitTerm& s) {
    T val;
    std::stringstream ss(s.str());
    ss >> val;
    return val;
}

inline JitTerm neg(const JitTerm& arg) {
    return JitTerm{"(-" + arg.str() + ")"};
}
inline JitTerm operator+(const JitTerm& lhs, const JitTerm& rhs) {
    if (lhs.str() == "0") {
        return rhs;
    }
    if (rhs.str() == "0") {
        return lhs;
    }
    if (is_number(lhs) && is_number(rhs)) {
        return JitTerm{std::to_string(as_number<int64_t>(lhs) + as_number<int64_t>(rhs))};
    }

    return JitTerm{"(" + lhs.str() + " + " + rhs.str() + ")"};
}

inline JitTerm operator-(const JitTerm& lhs, const JitTerm& rhs) {
    if (lhs.str() == "0") {
        return neg(rhs);
    }
    if (rhs.str() == "0") {
        return lhs;
    }
    if (is_number(lhs) && is_number(rhs)) {
        return JitTerm{std::to_string(as_number<int64_t>(lhs) - as_number<int64_t>(rhs))};
    }

    return JitTerm{"(" + lhs.str() + " - " + rhs.str() + ")"};
}

inline JitTerm operator*(const JitTerm& lhs, const JitTerm& rhs) {
    if (lhs.str() == "0" || rhs.str() == "0") {
        return JitTerm{"0"};
    }
    if (lhs.str() == "1") {
        return rhs;
    }
    if (rhs.str() == "1") {
        return lhs;
    }
    if (is_number(lhs) && is_number(rhs)) {
        return JitTerm{std::to_string(as_number<int64_t>(lhs) * as_number<int64_t>(rhs))};
    }
    return JitTerm{"(" + lhs.str() + " * " + rhs.str() + ")"};
}
inline JitTerm operator/(const JitTerm& lhs, const JitTerm& rhs) {
    if (rhs.str() == "1") {
        return lhs;
    }
    if (is_number(lhs) && is_number(rhs)) {
        return JitTerm{std::to_string(as_number<int64_t>(lhs) / as_number<int64_t>(rhs))};
    }
    return JitTerm{"(" + lhs.str() + " / " + rhs.str() + ")"};
}
inline JitTerm operator%(const JitTerm& lhs, const JitTerm& rhs) {
    return JitTerm{"(" + lhs.str() + " % " + rhs.str() + ")"};
}
inline JitTerm ternary(const JitTerm& condition, const JitTerm& true_expr, const JitTerm& false_expr) {
    return JitTerm{"(" + condition.str() + " ? " + true_expr.str() + " : " + false_expr.str() + ")"};
}
inline JitTerm isinf(const JitTerm& arg) {
    return JitTerm{"(isinf(" + arg.str() + "))"};
}
inline JitTerm exp(const JitTerm& arg) {
    return JitTerm{"(exp(" + arg.str() + "))"};
}
inline JitTerm erf(const JitTerm& arg) {
    return JitTerm{"(erf(" + arg.str() + "))"};
}
inline JitTerm tanh(const JitTerm& arg) {
    return JitTerm{"(tanh(" + arg.str() + "))"};
}
inline JitTerm log(const JitTerm& arg) {
    return JitTerm{"(log(" + arg.str() + "))"};
}
inline JitTerm operator"" _jit(const char* str, size_t /*unused*/) {
    return JitTerm{str};
}
inline JitTerm concat(const JitTerm& t1, const JitTerm& t2) {
    return JitTerm{t1.str() + t2.str()};
}

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

size_t extract_channel(ChannelName channel, const layout& l);
int get_channel_index(ChannelName channel_name, size_t rank, bool is_weights_fmt = false, bool is_grouped = false);
std::vector<ChannelName> get_default_channels_order(size_t rank, bool is_weights_fmt = false, bool is_grouped = false);

JitConstants make_layout_jit_constants(const std::string& name, const cldnn::layout& value, size_t shape_info_offset);
JitConstants make_type_jit_constants(const std::string& name, const ov::element::Type& value);
JitConstants make_indexing_jit_functions(const std::string& name, const layout& l);
JitConstants make_activation_jit_constants(activation_func activation_function,
                                           ov::element::Type_t out_dt,
                                           const std::string& suffix,
                                           bool use_type_parameter,
                                           bool disable_type_conversion);
JitConstants make_int4_packed_type_jit_constant(const std::string& macro_name, ov::element::Type type, size_t pack_size);
}  // namespace ov::intel_gpu
