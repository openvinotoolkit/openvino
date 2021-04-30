// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef _WIN32
    #include <Windows.h>
    #include <direct.h>
#else  // _WIN32
    #include <unistd.h>
    #include <dirent.h>
    #include <dlfcn.h>
#endif  // _WIN32

#include <string>
#include <vector>
#include <sys/stat.h>

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
#ifndef WIN32
    struct dirent *ent;
    DIR *dir = opendir(path.c_str());
    if (dir != nullptr)
    {
        std::unique_ptr<DIR, void (*)(DIR *)> closeGuard(dir, [](DIR *d)
        { closedir(d); });
        while ((ent = readdir(dir)) != NULL)
        {
            auto file = path + FileSeparator + std::string(ent->d_name);
            struct stat stat_path;
            stat(file.c_str(), &stat_path);
            if (!S_ISDIR(stat_path.st_mode) && file.find("_ngraph_frontend.so") != std::string::npos)
            {
                res.push_back(std::move(file));
            }
        }
    }
#else
    std::string searchPath = path + FileSeparator + "*_ngraph_frontend*.dll";
    WIN32_FIND_DATA fd;
    HANDLE handle = ::FindFirstFile(searchPath.c_str(), &fd);
    if (handle != INVALID_HANDLE_VALUE)
    {
        do
        {
            if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) )
            {
                res.push_back(path + FileSeparator + fd.cFileName);
            }
        } while(::FindNextFile(handle, &fd));
        ::FindClose(handle);
    }
#endif
    return res;
}

std::vector<PluginData> ngraph::frontend::loadPlugins(const std::string& dirName)
{
    auto files = listFiles(dirName);
    std::vector<PluginData> res;
    // std::cout << "Loading directory: " << dirName << "\n";
    for (const auto& file : files)
    {
        // std::cout << "Checking plugin: " << file << "\n";
        auto shared_object = DLOPEN(file);
        if (!shared_object)
        {
            continue;
        }

        PluginHandle guard([shared_object, file]()
                           {
                               // std::cout << "Closing plugin library " << file << std::endl;
                               DLCLOSE(shared_object);
                           });

        auto infoAddr = reinterpret_cast<void *(*)()>(DLSYM(shared_object, "GetAPIVersion"));
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

        auto creatorAddr = reinterpret_cast<void *(*)()>(DLSYM(shared_object, "GetFrontEndData"));
        if (!creatorAddr)
        {
            continue;
        }

        std::unique_ptr<FrontEndPluginInfo> fact{reinterpret_cast<FrontEndPluginInfo *>(creatorAddr())};

        res.push_back(PluginData(std::move(guard), std::move(*fact)));
    }
    return res;
}
