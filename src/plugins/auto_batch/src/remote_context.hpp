// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>

#include "openvino/runtime/remote_context.hpp"

namespace ov {
namespace autobatch_plugin {

class Plugin;
class CompiledModel;
class AutoBatchRemoteContext : public ov::RemoteContext {
public:
    friend class ov::autobatch_plugin::Plugin;
    friend class ov::autobatch_plugin::CompiledModel;
    AutoBatchRemoteContext(const ov::RemoteContext& remote_context = {}) : ov::RemoteContext(remote_context) {}
};

}  // namespace autobatch_plugin
}  // namespace ov