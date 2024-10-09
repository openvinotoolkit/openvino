// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/activation.hpp"
#include "intel_gpu/runtime/layout.hpp"
#include "openvino/core/type/element_type.hpp"

#include <string>

namespace ov {
namespace intel_gpu {
namespace ocl {

using namespace cldnn;

enum class ChannelName { X = 0, Y = 1, Z = 2, W = 3, U = 4, V = 5, FEATURE = 6, BATCH = 7, IFM = 8, OFM = 9, G = 10 };

struct JitConstant {
    std::string name;
    std::string value;
    JitConstant(const std::string& n, const std::string& v) : name(n), value(v) {}
};

template<typename T>
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

struct JitConstants : public std::vector<JitConstant> {
    void add(const JitConstant& constant) { push_back(constant); }
    void add(JitConstant&& constant) { push_back(constant); }

    template<typename... Args>
    void make(Args... args) { add(make_jit_constant(args...)); }

    void add(const std::vector<JitConstant>& constants) {
        insert(end(), constants.begin(), constants.end());
    }

    void merge(const JitConstants& jit) { add(jit); }

    void remove(std::string name) {
        erase(std::remove_if(begin(), end(), [=](const JitConstant& x) -> bool { return x.name == name; }), end());
    }

    JitConstants(std::initializer_list<JitConstant> values) : std::vector<JitConstant>(values) {}
    JitConstants() = default;
};

class JitTerm {
public:
    JitTerm() = default;
    explicit JitTerm(std::string text)
        : text(std::move(text)) {}

    std::string str() const { return text; }

    JitTerm gt(const JitTerm& rhs) const {
        JitTerm jit_term { "(" + text + ">" + rhs.str() + ")" };
        return jit_term;
    }

    JitTerm ge(const JitTerm& rhs) const {
        JitTerm jit_term {"(" + text + ">=" + rhs.str() + ")"};
        return jit_term;
    }

    JitTerm le(const JitTerm& rhs) const {
        JitTerm jit_term {"(" + text + "<=" + rhs.str() + ")"};
        return jit_term;
    }

    JitTerm eq(const JitTerm& rhs) const {
        JitTerm jit_term {"(" + text + "==" + rhs.str() + ")"};
        return jit_term;
    }

    template<typename... Args>
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
    template<typename... Args>
    JitTerm concat(const JitTerm& arg, Args... args, std::string separator) const {
        return concat(arg, concat(std::forward<Args>(args)..., separator), separator);
    }

    JitTerm concat(const JitTerm& t1, const JitTerm& t2, std::string separator) const {
        return JitTerm{t1.str() + separator + t2.str()};
    }

    JitTerm concat(const JitTerm& t1, std::string separator) const {
        return JitTerm{t1.str()};
    }

    JitTerm concat(std::string separator) const {
        return JitTerm{""};
    }

    std::string text;
};

inline JitTerm operator+(const JitTerm& lhs, const JitTerm& rhs) { return JitTerm{"(" + lhs.str() + " + " + rhs.str() + ")"}; }
inline JitTerm operator-(const JitTerm& lhs, const JitTerm& rhs) { return JitTerm{"(" + lhs.str() + " - " + rhs.str() + ")"}; }
inline JitTerm operator*(const JitTerm& lhs, const JitTerm& rhs) { return JitTerm{"(" + lhs.str() + " * " + rhs.str() + ")"}; }
inline JitTerm operator/(const JitTerm& lhs, const JitTerm& rhs) { return JitTerm{"(" + lhs.str() + " / " + rhs.str() + ")"}; }
inline JitTerm operator%(const JitTerm& lhs, const JitTerm& rhs) { return JitTerm{"(" + lhs.str() + " % " + rhs.str() + ")"}; }
inline JitTerm neg(const JitTerm& arg) { return JitTerm{"(-" + arg.str() + ")"}; }
inline  JitTerm ternary(const JitTerm& condition, const JitTerm& true_expr, const JitTerm& false_expr) {
    return JitTerm{"(" + condition.str() + " ? " + true_expr.str() + " : " + false_expr.str() + ")"};
}
inline JitTerm isinf(const JitTerm& arg) { return JitTerm{"(isinf(" + arg.str() + "))"}; }
inline JitTerm exp(const JitTerm& arg) { return JitTerm{"(exp(" + arg.str() + "))"}; }
inline JitTerm erf(const JitTerm& arg) { return JitTerm{"(erf(" + arg.str() + "))"}; }
inline JitTerm tanh(const JitTerm& arg) { return JitTerm{"(tanh(" + arg.str() + "))"}; }
inline JitTerm log(const JitTerm& arg) { return JitTerm{"(log(" + arg.str() + "))"}; }
inline JitTerm operator"" _jit(const char* str, size_t) { return JitTerm{str}; }
inline JitTerm concat(const JitTerm& t1, const JitTerm& t2) { return JitTerm{t1.str() + t2.str()}; }

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

    LayoutJitter(const layout& l, size_t shape_info_idx) {
        OPENVINO_ASSERT(!format::is_weights_format(l.format));
        make_definitions(l, shape_info_idx);
    }

    std::map<ChannelName, size_t> channels_map;

    std::string dim(ChannelName channel) const {
        return m_dims[channels_map.at(channel)].str();
    }

    std::string pad_l(ChannelName channel) const {
        return m_pad_lower[channels_map.at(channel)].str();
    }

    std::string pad_u(ChannelName channel) const {
        return m_pad_upper[channels_map.at(channel)].str();
    }

    std::string stride(ChannelName channel) const {
        return m_strides[channels_map.at(channel)].str();
    }

    std::string offset() const {
        return m_offset.str();
    }

private:
    void make_definitions(const layout& l, size_t shape_info_tensor_idx);
};

size_t extract_channel(ChannelName channel, const layout& l);

JitConstants make_layout_jit_constants(const std::string& name, const cldnn::layout& value, size_t shape_info_tensor_idx);
JitConstants make_type_jit_constants(const std::string& name, const ov::element::Type& value);
JitConstants make_indexing_jit_functions(const std::string& name, const layout& l);
JitConstants make_activation_jit_constants(activation_func activation_function,
                                           ov::element::Type_t out_dt,
                                           const std::string& suffix,
                                           bool use_type_parameter,
                                           bool disable_type_conversion);

}  // namespace ocl
}  // namespace intel_gpu
}  // namespace ov
