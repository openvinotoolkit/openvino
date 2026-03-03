// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "node/include/node_wrap.hpp"

#include "node/include/errors.hpp"
#include "node/include/type_validation.hpp"

NodeWrap::NodeWrap(const Napi::CallbackInfo& info) : Napi::ObjectWrap<NodeWrap>(info), _node{} {}

Napi::Function NodeWrap::get_class(Napi::Env env) {
    return DefineClass(env, "NodeWrap", {InstanceMethod("getName", &NodeWrap::get_name)});
}

void NodeWrap::set_node(const std::shared_ptr<ov::Node>& node) {
    _node = node;
}

std::shared_ptr<ov::Node> NodeWrap::get_node() const {
    return _node;
}

Napi::Value NodeWrap::get_name(const Napi::CallbackInfo& info) {
    std::vector<std::string> allowed_signatures;
    try {
        OPENVINO_ASSERT(ov::js::validate(info, allowed_signatures),
                        "'getName'",
                        ov::js::get_parameters_error_msg(info, allowed_signatures));
        return Napi::String::New(info.Env(), _node->get_name());
    } catch (const std::exception& e) {
        reportError(info.Env(), e.what());
        return info.Env().Undefined();
    }
}
