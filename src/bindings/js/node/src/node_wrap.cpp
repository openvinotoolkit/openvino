// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "node/include/node_wrap.hpp"

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
    if (_node->get_name() != "")
        return Napi::String::New(info.Env(), _node->get_name());
    else
        return Napi::String::New(info.Env(), "unknown");
}
