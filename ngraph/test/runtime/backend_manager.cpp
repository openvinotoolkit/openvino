//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <sstream>

#include "backend.hpp"
#include "backend_manager.hpp"
#include "ngraph/file_util.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

#ifdef NGRAPH_DYNAMIC_COMPONENTS_ENABLE
#ifdef _WIN32
#define CLOSE_LIBRARY(a) FreeLibrary(a)
#define DLSYM(a, b) GetProcAddress(a, b)
#define DLERROR() ""
#else
#define CLOSE_LIBRARY(a) dlclose(a)
#define DLSYM(a, b) dlsym(a, b)
string DLERROR()
{
    const char* error = dlerror();
    return error == nullptr ? "" : error;
}
#endif
#else
#define DLERROR() ""
#endif

unordered_map<string, runtime::BackendConstructor>& runtime::BackendManager::get_registry()
{
    static unordered_map<string, BackendConstructor> s_registered_backend;
    return s_registered_backend;
}

void runtime::BackendManager::register_backend(const string& name, BackendConstructor new_backend)
{
    get_registry()[name] = new_backend;
}

vector<string> runtime::BackendManager::get_registered_backends()
{
    vector<string> rc;
    for (const auto& p : get_registry())
    {
        rc.push_back(p.first);
    }
    for (const auto& p : get_registered_device_map())
    {
        if (find(rc.begin(), rc.end(), p.first) == rc.end())
        {
            rc.push_back(p.first);
        }
    }
    return rc;
}

shared_ptr<runtime::Backend> runtime::BackendManager::create_backend(std::string config)
{
    string type = config;
    string options;

    // strip off attributes, IE:CPU becomes IE
    auto colon = type.find(":");
    if (colon != type.npos)
    {
        options = type.substr(colon + 1);
        type = type.substr(0, colon);
    }

    auto& registry = get_registry();
    auto it = registry.find(type);
    string error;
#ifdef NGRAPH_DYNAMIC_COMPONENTS_ENABLE
    if (it == registry.end())
    {
        DL_HANDLE handle = open_shared_library(type);
        if (!handle)
        {
            error = DLERROR();
        }
        else
        {
            DLERROR(); // Clear any pending errors
            string register_function_name =
                string("ngraph_register_") + to_lower(type) + "_backend";
            auto register_function =
                reinterpret_cast<void (*)()>(DLSYM(handle, register_function_name.c_str()));
            if (register_function)
            {
                register_function();
                it = registry.find(type);
            }
            else
            {
                error = DLERROR();
                CLOSE_LIBRARY(handle);
                stringstream ss;
                ss << "Failed to find symbol 'get_backend_constructor_pointer' in backend library."
                   << endl;
                if (error.size() > 0)
                {
                    ss << "\nError: " << error;
                }
                error = ss.str();
            }
        }
    }
#endif

    if (it == registry.end())
    {
        stringstream ss;
        ss << "Backend '" << type << "' not registered.";
        if (error.size() > 0)
        {
            ss << "\n  Error: " << DLERROR();
        }
        throw runtime_error(ss.str());
    }
    return it->second(options);
}

DL_HANDLE runtime::BackendManager::open_shared_library(string type)
{
    DL_HANDLE handle = nullptr;
#ifdef NGRAPH_DYNAMIC_COMPONENTS_ENABLE
    string lib_prefix = SHARED_LIB_PREFIX;
    string lib_suffix = SHARED_LIB_SUFFIX;

    // strip off attributes, IE:CPU becomes IE
    auto colon = type.find(":");
    if (colon != type.npos)
    {
        type = type.substr(0, colon);
    }

    string library_name = lib_prefix + to_lower(type) + "_backend" + lib_suffix;
    string my_directory =
        file_util::get_directory(Backend::get_backend_shared_library_search_directory());
    string library_path = file_util::path_join(my_directory, library_name);
#ifdef _WIN32
    SetDllDirectoryA((LPCSTR)my_directory.c_str());
    handle = LoadLibraryA(library_path.c_str());
#elif defined(__APPLE__) || defined(__linux__)
    DLERROR(); // Clear any pending errors
    handle = dlopen(library_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
#else
#error "Unsupported OS"
#endif
    string error = DLERROR();
    if (!handle)
    {
        stringstream ss;
        ss << "Unable to find backend '" << type << "' as file '" << library_path << "'";
        if (error.size() > 0)
        {
            ss << "\nOpen error message '" << error << "'";
        }
        throw runtime_error(ss.str());
    }
#endif
    return handle;
}

map<string, string> runtime::BackendManager::get_registered_device_map()
{
    map<string, string> rc;
#ifdef NGRAPH_DYNAMIC_COMPONENTS_ENABLE
    string my_directory =
        file_util::get_directory(Backend::get_backend_shared_library_search_directory());
    vector<string> backend_list;

    auto f = [&](const string& file, bool is_dir) {
        if (!is_dir)
        {
            string name = file_util::get_file_name(file);
            string backend_name;
            if (is_backend_name(name, backend_name))
            {
                rc.insert({to_upper(backend_name), file});
            }
        }
    };
    file_util::iterate_files(my_directory, f, false, true);
#endif
    return rc;
}

bool runtime::BackendManager::is_backend_name(const string& file, string& backend_name)
{
    bool rc = false;
    string name = file_util::get_file_name(file);
    string lib_prefix = SHARED_LIB_PREFIX;
    string lib_suffix = SHARED_LIB_SUFFIX;
    if ((name.size() > lib_prefix.size() + lib_suffix.size()) &
        !name.compare(0, lib_prefix.size(), lib_prefix))
    {
        if (!name.compare(name.size() - lib_suffix.size(), lib_suffix.size(), lib_suffix))
        {
            auto pos = name.find("_backend");
            if (pos != name.npos)
            {
                backend_name = name.substr(lib_prefix.size(), pos - lib_prefix.size());
                rc = true;
            }
        }
    }
    return rc;
}
