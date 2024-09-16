// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/layout.hpp"

#include <string>

namespace ov {
namespace intel_gpu {
namespace ocl {

using namespace cldnn;

enum class DataChannelName { X = 0, Y = 1, Z = 2, W = 3, U = 4, V = 5, FEATURE = 6, BATCH = 7, COUNT = 8 };
enum class WeightsChannelName { X = 0, Y = 1, Z = 2, IFM = 3, OFM = 4, G = 5, COUNT = 6 };

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

JitConstants make_jit_constants(const std::string& name, const cldnn::layout& value);
JitConstants make_jit_constants(const std::string& name, const ov::element::Type& value);
JitConstants make_indexing_jit_functions(const std::string& name, const layout& l);

}  // namespace ocl
}  // namespace intel_gpu
}  // namespace ov
