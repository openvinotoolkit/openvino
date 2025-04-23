// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/openvino.hpp"
#include <napi.h>

class NodeWrap : public Napi::ObjectWrap<NodeWrap> {
public:
  // Constructor that initializes node_ from Napi::CallbackInfo
  explicit NodeWrap(const Napi::CallbackInfo &info);

  // Delete default constructor to prevent creating with default values
  NodeWrap() = delete;

  // Member functions
  Napi::Value get_name(const Napi::CallbackInfo &info) const;

  // Static function to get the class constructor
  static Napi::Function get_class(Napi::Env env);

  // Static function to create a new instance
  static Napi::Object New(Napi::Env env, const ov::Node &node);

private:
  // Internal node object (using a reference to ensure it's never null)
  const ov::Node &node_;
};