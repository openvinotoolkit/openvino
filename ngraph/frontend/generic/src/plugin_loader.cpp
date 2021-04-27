// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef _WIN32
#include <direct.h>
#define rmdir(dir) _rmdir(dir)
#else  // _WIN32
#include <unistd.h>
#include <dirent.h>
#include <dlfcn.h>

const char FileSeparator[] = "/";
#endif  // _WIN32

#include <string>
#include <vector>
#include <sys/stat.h>

#include "plugin_loader.hpp"

// TODO: change to std::filesystem for C++17
static std::vector<std::string> listFiles(const std::string& path) {
    struct dirent *ent;
    std::vector<std::string> res;
    DIR *dir = opendir(path.c_str());
    if (dir != nullptr) {
        std::unique_ptr<DIR, void(*)(DIR*)> closeGuard(dir, [](DIR* d) { closedir(d); });
        while ((ent = readdir(dir)) != NULL) {
            auto file = path + FileSeparator + std::string(ent->d_name);
            struct stat stat_path;
            stat(file.c_str(), &stat_path);
            if (!S_ISDIR(stat_path.st_mode)) {
                res.push_back(std::move(file));
            }
        }
    }
    return res;
}

namespace ngraph {
namespace frontend {
std::vector<PluginFactoryValue> loadPlugins(const std::string& dirName) {
    auto files = listFiles(dirName);
    std::vector<PluginFactoryValue> res;
    for (const auto& file : files) {
#ifdef _WIN32
        throw std::runtime_error("Not yet implemented");
#else
        void* shared_object = dlopen(file.c_str(), RTLD_LAZY);
        if (shared_object == nullptr)
            continue;

        void* infoAddr = dlsym(shared_object, "GetPluginInfo");
        if (infoAddr == nullptr)
            continue;

        std::function<PluginInfo()> infoFunc = reinterpret_cast<PluginInfo(*)()>(infoAddr);
        PluginInfo info = infoFunc();

        // TODO: validate plugin version compatibility here

        void* factoryAddr = dlsym(shared_object, "GetFrontEndFactory");
        if (factoryAddr == nullptr)
            continue;

        std::function<FrontEndFactory()> factoryFunc =
                reinterpret_cast<FrontEndFactory(*)()>(factoryAddr);
        FrontEndFactory creator = factoryFunc();
        res.emplace_back(info, creator);
#endif
    }
    return res;
}
}  // namespace frontend
}  // namespace ngraph
