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

namespace ngraph {
namespace frontend {

std::vector<PluginData> loadPlugins(const std::string& dirName) {
    auto files = listFiles(dirName);
    std::vector<PluginData> res;
    for (const auto& file : files) {
#ifdef _WIN32
        auto hinstLib = LoadLibrary(TEXT(file.c_str()));
        if (hinstLib != NULL)
        {
            PluginHandle guard([hinstLib, file]() {
                FreeLibrary(hinstLib);
            });

            std::function<void*()> infoAddr = reinterpret_cast<void*(*)()>(GetProcAddress(hinstLib, "GetPluginInfo"));
            if (NULL == infoAddr) {
                continue;
            }

            PluginInfo* plugInfo = reinterpret_cast<PluginInfo*>(infoAddr());

            std::function<void*()> creatorAddr = reinterpret_cast<void*(*)()>(GetProcAddress(hinstLib, "GetFrontEndFactory"));
            if (NULL == creatorAddr) {
                continue;
            }

            FrontEndFactory& fact = (*reinterpret_cast<FrontEndFactory*>(creatorAddr()));

            PluginData data(*plugInfo, std::move(fact), std::move(guard));
            res.push_back(std::move(data));
        }
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
        // TODO: call DLClose
#endif
    }
    return res;
}
}  // namespace frontend
}  // namespace ngraph
