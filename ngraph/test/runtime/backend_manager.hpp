// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#define DL_HANDLE HMODULE
#else
#define DL_HANDLE void*
#endif

#include "backend_visibility.hpp"
#include "ngraph/ngraph_visibility.hpp"

namespace ngraph
{
    namespace runtime
    {
        class Backend;
        class BackendManager;
        using BackendConstructor =
            std::function<std::shared_ptr<ngraph::runtime::Backend>(const std::string& config)>;
    }
}

class ngraph::runtime::BackendManager
{
    friend class Backend;

public:
    /// \brief Used by build-in backends to register their name and constructor.
    ///    This function is not used if the backend is build as a shared library.
    /// \param name The name of the registering backend in UPPER CASE.
    /// \param backend_constructor A BackendConstructor which will be called to
    ////     construct an instance of the registered backend.
    static BACKEND_API void register_backend(const std::string& name,
                                             BackendConstructor backend_constructor);

    /// \brief Query the list of registered devices
    /// \returns A vector of all registered devices.
    static BACKEND_API std::vector<std::string> get_registered_backends();

private:
    static void initialize_backends();
    static std::shared_ptr<runtime::Backend> create_backend(std::string type);
    static std::unordered_map<std::string, BackendConstructor>& get_registry();

    static std::unordered_map<std::string, BackendConstructor> s_registered_backend;

    static DL_HANDLE open_shared_library(std::string type);
    static std::map<std::string, std::string> get_registered_device_map();
    static bool is_backend_name(const std::string& file, std::string& backend_name);
};
