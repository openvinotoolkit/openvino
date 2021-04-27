// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <frontend_manager/ifrontend_manager.hpp>

namespace ngraph {
namespace frontend {

    std::vector<PluginFactoryValue> loadPlugins(const std::string& dirName);

}  // namespace frontend
}  // namespace ngraph
