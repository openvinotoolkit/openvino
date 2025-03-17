// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef _WIN32
#    include <cxxabi.h>
#endif

#include <memory>
#include <openvino/cc/pass/itt.hpp>

#include "atomic_guard.hpp"
#include "openvino/pass/manager.hpp"

using namespace std;

ov::pass::PassBase::PassBase() : m_property(), m_name(), m_pass_config(std::make_shared<PassConfig>()) {}

ov::pass::PassBase::~PassBase() = default;

bool ov::pass::PassBase::get_property(const PassPropertyMask& prop) const {
    return m_property.is_set(prop);
}

void ov::pass::PassBase::set_property(const PassPropertyMask& prop, bool value) {
    if (value) {
        m_property.set(prop);
    } else {
        m_property.clear(prop);
    }
}

std::string ov::pass::PassBase::get_name() const {
    if (m_name.empty()) {
        const PassBase* p = this;
        std::string pass_name = typeid(*p).name();
#ifndef _WIN32
        int status;
        std::unique_ptr<char, void (*)(void*)> demangled_name(
            abi::__cxa_demangle(pass_name.c_str(), nullptr, nullptr, &status),
            std::free);
        pass_name = demangled_name.get();
#endif
        return pass_name;
    } else {
        return m_name;
    }
}

void ov::pass::PassBase::set_callback(const param_callback& callback) {
    m_pass_config->set_callback(callback);
}

// The symbols are requiered to be in cpp file to workaround RTTI issue on Android LLVM

ov::pass::ModelPass::~ModelPass() = default;
