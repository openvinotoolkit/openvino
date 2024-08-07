// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/file_utils.hpp"

#include "openvino/runtime/core.hpp"
#include "openvino/runtime/iplugin.hpp"
#include "openvino/util/shared_object.hpp"

namespace ov {
namespace npuw {
namespace tests {

template <class T>
std::function<T> make_std_function(const std::shared_ptr<void> so, const std::string& functionName) {
    std::function<T> ptr(reinterpret_cast<T*>(ov::util::get_symbol(so, functionName.c_str())));
    return ptr;
}

std::shared_ptr<void> reg_plugin(ov::Core& core, std::shared_ptr<ov::IPlugin>& plugin);

template <typename T>
std::shared_ptr<void> reg_plugin(ov::Core& core, const std::shared_ptr<T>& plugin) {
    auto base_plugin_ptr = std::dynamic_pointer_cast<ov::IPlugin>(plugin);
    return reg_plugin(core, base_plugin_ptr);
}
}
} // namespace npuw
} // namespace tests
