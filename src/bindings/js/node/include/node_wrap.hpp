// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <napi.h>

#include "openvino/core/node.hpp"

class NodeWrap : public Napi::ObjectWrap<NodeWrap> {
public:
    /**
     * @brief Constructs NodeWrap from the Napi::CallbackInfo.
     * @param info contains passed arguments. Can be empty.
     */
    NodeWrap(const Napi::CallbackInfo& info);

    /**
     * @brief Defines a Javascript Node class with constructor, static and instance properties and methods.
     * @param env The environment in which to construct a JavaScript class.
     * @return Napi::Function representing the constructor function for the Javascript Node class.
     */
    static Napi::Function get_class(Napi::Env env);

    /** @return Napi::String containing a node unique name. */
    Napi::Value get_name(const Napi::CallbackInfo& info);

    std::shared_ptr<ov::Node> get_node() const;

    void set_node(const std::shared_ptr<ov::Node>& node);

private:
    std::shared_ptr<ov::Node> _node;
};
