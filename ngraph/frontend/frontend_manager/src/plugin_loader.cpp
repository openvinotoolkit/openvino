// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin_loader.hpp"

#include <string>
#include <vector>

#include "openvino/util/file_util.hpp"
#include "openvino/util/shared_object.hpp"

using namespace ov;
using namespace ov::frontend;

// TODO: change to std::filesystem for C++17
static std::vector<std::string> list_files(const std::string& path) {
    std::vector<std::string> res;
    try {
        ov::util::iterate_files(
            path,
            [&res](const std::string& file, bool is_dir) {
                if (!is_dir && file.find("_ngraph_frontend") != std::string::npos) {
#ifdef _WIN32
                    std::string ext = ".dll";
#elif defined(__APPLE__)
                    std::string ext = ".dylib";
#else
                    std::string ext = ".so";
#endif
                    if (file.find(ext) != std::string::npos) {
                        res.push_back(file);
                    }
                }
            },
            // ilavreno: this is current solution for static runtime
            // since frontends are still dynamic libraries and they are located in
            // a different folder with compare to frontend_manager one (in ./lib)
            // we are trying to use recursive search. Can be reverted back in future
            // once the frontends are static too.
            true,
            true);
    } catch (...) {
        // Ignore exceptions
    }
    return res;
}

std::vector<PluginData> ov::frontend::load_plugins(const std::string& dir_name) {
    auto files = list_files(dir_name);
    std::vector<PluginData> res;
    for (const auto& file : files) {
        auto shared_object = ov::util::load_shared_object(file.c_str());
        if (!shared_object) {
            continue;
        }

        auto info_addr = reinterpret_cast<void* (*)()>(ov::util::get_symbol(shared_object, "GetAPIVersion"));
        if (!info_addr) {
            continue;
        }
        FrontEndVersion plug_info{reinterpret_cast<FrontEndVersion>(info_addr())};

        if (plug_info != OV_FRONTEND_API_VERSION) {
            // Plugin has incompatible API version, do not load it
            continue;
        }

        auto creator_addr = reinterpret_cast<void* (*)()>(ov::util::get_symbol(shared_object, "GetFrontEndData"));
        if (!creator_addr) {
            continue;
        }

        std::unique_ptr<FrontEndPluginInfo> fact{reinterpret_cast<FrontEndPluginInfo*>(creator_addr())};

        res.push_back(PluginData(shared_object, std::move(*fact)));
    }
    return res;
}
