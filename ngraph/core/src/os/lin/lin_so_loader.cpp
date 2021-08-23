// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <dlfcn.h>

#include <iostream>
#include <memory>

#include "ngraph/check.hpp"
#include "ngraph/file_util.hpp"
#include "openvino/core/so_loader.hpp"

namespace ov {

class SOLoader::Impl {
private:
    void* shared_object = nullptr;

public:
    explicit Impl(const char* pluginName) {
        shared_object = dlopen(pluginName, RTLD_NOW);

        if (shared_object == nullptr)
            NGRAPH_CHECK(false, "Cannot load library '", pluginName, "': ", dlerror());
    }

#ifdef ENABLE_UNICODE_PATH_SUPPORT
    explicit Impl(const wchar_t* pluginName) : Impl(ngraph::file_util::wstring_to_string(pluginName).c_str()) {}
#endif  // ENABLE_UNICODE_PATH_SUPPORT

    ~Impl() {
        if (0 != dlclose(shared_object)) {
            std::cerr << "dlclose failed: " << dlerror() << std::endl;
        }
    }

    /**
     * @brief Searches for a function symbol in the loaded module
     * @param symbolName Name of the function to find
     * @return A pointer to the function if found
     * @throws Exception if the function is not found
     */
    void* get_symbol(const char* symbolName) const {
        void* procAddr = nullptr;

        procAddr = dlsym(shared_object, symbolName);
        if (procAddr == nullptr)
            NGRAPH_CHECK(false, "dlSym cannot locate method '", symbolName, "': ", dlerror());
        return procAddr;
    }
};

#ifdef ENABLE_UNICODE_PATH_SUPPORT
SOLoader::SOLoader(const std::wstring& pluginName) {
    _impl = std::make_shared<Impl>(pluginName.c_str());
}
#endif

SOLoader::SOLoader(const std::string& pluginName) {
    _impl = std::make_shared<Impl>(pluginName.c_str());
}

SOLoader::~SOLoader() = default;

void* SOLoader::get_symbol(const std::string& symbolName) const {
    if (_impl == nullptr) {
        NGRAPH_CHECK(false, "SOLoader is not initialized");
    }
    return _impl->get_symbol(symbolName.c_str());
}

}  // namespace ov

