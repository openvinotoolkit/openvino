// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <napi.h>

#include "openvino/core/node_output.hpp"

template <class NodeType>
class Output : public Napi::ObjectWrap<Output<NodeType>> {};

template <>
class Output<ov::Node> : public Napi::ObjectWrap<Output<ov::Node>> {
public:
    Output(const Napi::CallbackInfo& info);

    /**
     * @brief Defines a Javascript Output class with constructor, static and instance properties and methods.
     * @param env The environment in which to construct a JavaScript class.
     * @return Napi::Function representing the constructor function for the Javascript Output class.
     */
    static Napi::Function get_class(Napi::Env env);

    ov::Output<ov::Node> get_output() const;

    static Napi::Object wrap(Napi::Env env, ov::Output<ov::Node> output);

    Napi::Value get_shape(const Napi::CallbackInfo& info);

    Napi::Value get_partial_shape(const Napi::CallbackInfo& info);

    Napi::Value get_shape_data(const Napi::CallbackInfo& info);

    Napi::Value get_any_name(const Napi::CallbackInfo& info);

private:
    ov::Output<ov::Node> _output;
};

template <>
class Output<const ov::Node> : public Napi::ObjectWrap<Output<const ov::Node>> {
public:
    Output(const Napi::CallbackInfo& info);

    /**
     * @brief Defines a Javascript Output class with constructor, static and instance properties and methods.
     * @param env The environment in which to construct a JavaScript class.
     * @return Napi::Function representing the constructor function for the Javascript Output class.
     */
    static Napi::Function get_class(Napi::Env env);

    ov::Output<const ov::Node> get_output() const;

    static Napi::Object wrap(Napi::Env env, ov::Output<const ov::Node> output);

    Napi::Value get_shape(const Napi::CallbackInfo& info);

    Napi::Value get_partial_shape(const Napi::CallbackInfo& info);

    Napi::Value get_shape_data(const Napi::CallbackInfo& info);

    Napi::Value get_any_name(const Napi::CallbackInfo& info);

private:
    ov::Output<const ov::Node> _output;
};
