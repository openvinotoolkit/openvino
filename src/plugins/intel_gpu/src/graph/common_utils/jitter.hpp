// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <sstream>
#include <string>
#include <utility>

#include "common_utils/dispatch_utils.hpp"
#include "common_utils/jit_term.hpp"
#include "intel_gpu/runtime/layout.hpp"

namespace ov::intel_gpu {

class CodeBuilder {
    std::ostringstream code;
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
    CodeBuilder& add_line(const std::string& line) {
        code << line << "\n";
        return *this;
    }

    CodeBuilder& decoration_macro(const std::string& name,
                                  const std::string& prefix,
                                  const std::string& postfix,
                                  const std::string& name_prefix = std::string()) {
        code << "#define " << name << "(name) " << prefix << " " + name_prefix + "_##" + "name" << (postfix.empty() ? "" : "##_") << postfix << '\n';
        return register_macro(name);
    }

    CodeBuilder& value_macro(const std::string& name, const std::string& value) {
        code << "#define " << name << " " << value << '\n';
        return register_macro(name.substr(0, name.find('(')));
    }

    CodeBuilder& undef_macro(const std::string& name) {
        code << "#undef " << name.substr(0, name.find('(')) << '\n';
        return unregister_macro(name.substr(0, name.find('(')));
    }

    std::string str() {
        code << '\n';
        return code.str();
    }
};

struct JitConstant {
    std::string name;
    std::string value;
    JitConstant(std::string n, std::string v) : name(std::move(n)), value(std::move(v)) {}
};

template <typename T>
JitConstant make_jit_constant(const std::string& name, T value) {
    return JitConstant(name, to_code_string(value));
}
template <typename T>
inline JitConstant make_jit_constant(const JitTerm& name, T value) {
    return JitConstant(name.str(), to_code_string(value));
}

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

size_t extract_channel(ChannelName channel, const cldnn::layout& l);
int get_channel_index(ChannelName channel_name, size_t rank, bool is_weights_fmt = false, bool is_grouped = false);
std::vector<ChannelName> get_default_channels_order(size_t rank, bool is_weights_fmt = false, bool is_grouped = false);

}  // namespace ov::intel_gpu
