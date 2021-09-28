// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file for the Parameter class
 * @file openvino/runtime/parameter.hpp
 */
#pragma once

#include "openvino/core/except.hpp"
#include "openvino/core/shared_any.hpp"
#include "openvino/runtime/common.hpp"

namespace ov {
namespace runtime {
class Core;
class ExecutableNetwork;
class RemoteContext;

/**
 * @brief This class represents an object to work with different parameters loaded from inference plugins
 *
 */
class Parameter {
    std::shared_ptr<void> _so;
    SharedAny _impl;

    Parameter(const std::shared_ptr<void>& so, const SharedAny& impl) : _so{so}, _impl{impl} {}

    friend class ov::runtime::Core;
    friend class ov::runtime::ExecutableNetwork;
    friend class ov::runtime::RemoteContext;

    template <typename T>
    bool is() const {
        return _impl.is<T>();
    }

    template <typename T>
    T& as() const {
        return _impl.as<T>();
    }
};

/**
 * @brief This type of map is commonly used to return set of loaded from inference plugin
 */
using ParamMap = std::map<std::string, Parameter>;
}  // namespace runtime
}  // namespace ov
