// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "register_in_ov.hpp"

#include <filesystem>

namespace ov {
namespace npuw {
namespace tests {

std::shared_ptr<void> reg_plugin(ov::Core& core, std::shared_ptr<ov::IPlugin>& plugin) {
    std::string mock_engine_path = ov::util::make_plugin_library_name(ov::test::utils::getExecutableDirectory(),
                                                                      std::string("mock_engine") + OV_BUILD_POSTFIX);
    std::string plugin_path = ov::util::make_plugin_library_name(
            ov::test::utils::getExecutableDirectory(),
            std::string("mock_engine_") + plugin->get_device_name() + OV_BUILD_POSTFIX);
    if (!std::filesystem::is_regular_file(plugin_path)) {
        std::filesystem::copy(mock_engine_path, plugin_path);
    }

    auto so = ov::util::load_shared_object(plugin_path.c_str());

    std::function<void(ov::IPlugin*)> injectProxyEngine =
            make_std_function<void(ov::IPlugin*)>(so, "InjectPlugin");

    injectProxyEngine(plugin.get());
    core.register_plugin(plugin_path, plugin->get_device_name());

    return so;
}
}
} // namespace npuw
} // namespace tests
