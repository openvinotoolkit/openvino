// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef _WIN32
#else
#    include <cxxabi.h>
#endif

#include "ngraph/pass/pass.hpp"
#include "openvino/pass/manager.hpp"

using namespace std;

OPENVINO_RTTI_DEFINITION(ov::pass::FunctionPass, "ov::pass::FunctionPass", 0);

ov::pass::PassBase::PassBase() : m_property(), m_pass_config(std::make_shared<PassConfig>()) {}

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
        pass_name = abi::__cxa_demangle(pass_name.c_str(), nullptr, nullptr, &status);
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

ov::pass::FunctionPass::~FunctionPass() = default;

OPENVINO_SUPPRESS_DEPRECATED_START

OPENVINO_RTTI_DEFINITION(ngraph::pass::NodePass, "ngraph::pass::NodePass", 0);

ngraph::pass::NodePass::~NodePass() = default;
