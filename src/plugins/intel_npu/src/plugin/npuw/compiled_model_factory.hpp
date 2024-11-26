// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <optional>

#include "common.hpp"
#include "openvino/openvino.hpp"
#include "openvino/runtime/icompiled_model.hpp"
#include "openvino/runtime/so_ptr.hpp"

namespace intel_npu {
class Plugin;
}

namespace ov {
namespace npuw {

class CompiledModelFactory {
public:
    static std::shared_ptr<ov::ICompiledModel> create(const std::shared_ptr<ov::Model>& model,
                                                      const std::shared_ptr<const ov::IPlugin>& plugin,
                                                      const ov::AnyMap& properties);
};

} // namespace npuw
} // namespace ov
