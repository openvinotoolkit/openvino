// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "node/include/node_wrap.hpp"

#include "node/include/errors.hpp"
#include "node/include/type_validation.hpp"

Napi::FunctionReference NodeWrap::constructor;

Napi::Function NodeWrap::get_class(Napi::Env env) {
    return DefineClass(env, "NodeWrap", {InstanceMethod("getName", &NodeWrap::get_name)});
}

NodeWrap::NodeWrap(const Napi::CallbackInfo& info) : Napi::ObjectWrap<NodeWrap>(info) {
    Napi::Env env = info.Env();
    std::vector<std::string> allowed_signatures;

    try {
        if (ov::js::validate<Napi::External<ov::Node>>(info, allowed_signatures)) {
            node = std::shared_ptr<ov::Node>(info[0].As<Napi::External<ov::Node>>().Data());

        } else if (ov::js::validate<>(info, allowed_signatures)) {
            // Use default node initialization
            node = std::make_shared<ov::Node>();
        } else {
            OPENVINO_THROW("'NodeWrap'", ov::js::get_parameters_error_msg(info, allowed_signatures));
        }
    } catch (std::runtime_error& err) {
        reportError(info.Env(), err.what());

        return info.Env().Undefined();
    }
}

Napi::Value NodeWrap::get_name(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    return Napi::String::New(env, node->get_name());
}
