// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef _WIN32
#include <Windows.h>
#include <direct.h>
#else // _WIN32
#include <dirent.h>
#include <dlfcn.h>
#include <unistd.h>
#endif // _WIN32

#include <string>
#include <sys/stat.h>
#include <vector>
#include "ngraph/file_util.hpp"

#include "plugin_loader.hpp"

using namespace ngraph;
using namespace ngraph::frontend;

#ifdef WIN32
#define DLOPEN(fileStr) LoadLibrary(TEXT(fileStr.c_str()))
#define DLSYM(obj, func) GetProcAddress(obj, func)
#define DLCLOSE(obj) FreeLibrary(obj)
#else
#define DLOPEN(fileStr) dlopen(file.c_str(), RTLD_LAZY)
#define DLSYM(obj, func) dlsym(obj, func)
#define DLCLOSE(obj) dlclose(obj)
#endif

// TODO: change to std::filesystem for C++17
static std::vector<std::string> listFiles(const std::string& path)
{
    std::vector<std::string> res;
    try
    {
        ngraph::file_util::iterate_files(
            path,
            [&res](const std::string& file, bool is_dir) {
                if (!is_dir && file.find("_ngraph_frontend") != std::string::npos)
                {
#ifdef _WIN32
                    std::string ext = ".dll";
#elif defined(__APPLE__)
                    std::string ext = ".dylib";
#else
                    std::string ext = ".so";
#endif
                    if (file.find(ext) != std::string::npos)
                    {
                        res.push_back(file);
                    }
                }
            },
            false,
            true);
    }
    catch (...)
    {
        // Ignore exceptions
    }
    return res;
}

std::vector<PluginData> ngraph::frontend::loadPlugins(const std::string& dirName)
{
    auto files = listFiles(dirName);
    std::vector<PluginData> res;
    for (const auto& file : files)
    {
        auto shared_object = DLOPEN(file);
        if (!shared_object)
        {
            continue;
        }

        PluginHandle guard([shared_object, file]() {
            // std::cout << "Closing plugin library " << file << std::endl;
            DLCLOSE(shared_object);
        });

        auto infoAddr = reinterpret_cast<void* (*)()>(DLSYM(shared_object, "GetAPIVersion"));
        if (!infoAddr)
        {
            continue;
        }
        FrontEndVersion plugInfo{reinterpret_cast<FrontEndVersion>(infoAddr())};

        if (plugInfo != OV_FRONTEND_API_VERSION)
        {
            // Plugin has incompatible API version, do not load it
            continue;
        }

        auto creatorAddr = reinterpret_cast<void* (*)()>(DLSYM(shared_object, "GetFrontEndData"));
        if (!creatorAddr)
        {
            continue;
        }

        std::unique_ptr<FrontEndPluginInfo> fact{
            reinterpret_cast<FrontEndPluginInfo*>(creatorAddr())};

        res.push_back(PluginData(std::move(guard), std::move(*fact)));
    }
    return res;
}
