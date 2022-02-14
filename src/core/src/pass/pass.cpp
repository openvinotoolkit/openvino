// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef _WIN32
#else
#    include <cxxabi.h>
#endif

#include <memory>

#include "atomic_guard.hpp"
#include "ngraph/pass/pass.hpp"
#include "openvino/pass/manager.hpp"

using namespace std;

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

namespace {
class RunLocker {
public:
    RunLocker(bool& flag) : m_flag(flag) {
        OPENVINO_ASSERT(m_flag == false,
                        "Cycle detected. run_on_model() or run_on_function() method should be overridden.");
        m_flag = true;
    }
    ~RunLocker() {
        m_flag = false;
    }

private:
    bool& m_flag;
};
}  // namespace

// The symbols are requiered to be in cpp file to workaround RTTI issue on Android LLVM

ov::pass::ModelPass::~ModelPass() = default;

OPENVINO_SUPPRESS_DEPRECATED_START

bool ov::pass::ModelPass::run_on_model(const std::shared_ptr<ov::Model>& m) {
    RunLocker locked(call_on_model);
    OPENVINO_ASSERT(!call_on_function,
                    "Cycle detected. run_on_model() or run_on_function() method should be overridden.");
    bool sts = run_on_function(m);
    return sts;
}

bool ov::pass::ModelPass::run_on_function(std::shared_ptr<ov::Model> m) {
    RunLocker locked(call_on_function);
    OPENVINO_ASSERT(!call_on_model, "Cycle detected. run_on_model() or run_on_function() method should be overridden.");
    bool sts = run_on_model(m);
    return sts;
}

NGRAPH_RTTI_DEFINITION(ngraph::pass::NodePass, "ngraph::pass::NodePass", 0);

ngraph::pass::NodePass::~NodePass() = default;
