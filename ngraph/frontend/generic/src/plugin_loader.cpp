// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef _WIN32
#include <Windows.h>
#include <direct.h>
const char FileSeparator[] = "\\";
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
    std::vector<std::string> res;
#ifndef WIN32
    struct dirent *ent;
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
#else
    std::string searchPath = path + FileSeparator + "*_frontend*.dll";
    WIN32_FIND_DATA fd;
    HANDLE handle = ::FindFirstFile(searchPath.c_str(), &fd);
    if (handle != INVALID_HANDLE_VALUE) {
        do {
            if(!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) ) {
                res.push_back(path + FileSeparator + fd.cFileName);
            }
        } while(::FindNextFile(handle, &fd));
        ::FindClose(handle);
    }
#endif
    return res;
}

#ifdef WIN32
#define DLOPEN(fileStr) LoadLibrary(TEXT(fileStr.c_str()))
#define DLSYM(obj, func) GetProcAddress(obj, func)
#define DLCLOSE(obj) FreeLibrary(obj)
#else
#define DLOPEN(fileStr) dlopen(file.c_str(), RTLD_LAZY)
#define DLSYM(obj, func) dlsym(obj, func)
#define DLCLOSE(obj) dlclose(obj)
#endif

namespace ngraph {
namespace frontend {

std::vector<PluginData> loadPlugins(const std::string& dirName) {
    auto files = listFiles(dirName);
    std::vector<PluginData> res;
    // std::cout << "Loading directory: " << dirName << "\n";
    for (const auto& file : files) {
        // std::cout << "Checking plugin: " << file << "\n";
        auto shared_object = DLOPEN(file);
        if (!shared_object)
            continue;

        PluginHandle guard([shared_object, file]() {
            // std::cout << "Closing plugin library " << file << std::endl;
            DLCLOSE(shared_object);
        });

        auto infoAddr = reinterpret_cast<void*(*)()>(DLSYM(shared_object, "GetPluginInfo"));
        if (!infoAddr) {
            continue;
        }

        std::unique_ptr<PluginInfo> plugInfo { reinterpret_cast<PluginInfo*>(infoAddr()) };

        auto creatorAddr = reinterpret_cast<void*(*)()>(DLSYM(shared_object, "GetFrontEndFactory"));
        if (!creatorAddr) {
            continue;
        }

        std::unique_ptr<FrontEndFactory> fact { reinterpret_cast<FrontEndFactory*>(creatorAddr()) };

        PluginData data(*plugInfo, std::move(*fact), std::move(guard));
        res.push_back(std::move(data));
    }
    return res;
}
}  // namespace frontend
}  // namespace ngraph
