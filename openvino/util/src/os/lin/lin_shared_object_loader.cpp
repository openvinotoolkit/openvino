// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <dlfcn.h>

#include <iostream>
#include <sstream>

#include "openvino/util/file_util.hpp"
#include "openvino/util/shared_object.hpp"
#include "openvino/util/so_loader.hpp"

namespace ov {
namespace util {
std::shared_ptr<void> load_shared_object(const char* path) {
    auto shared_object = std::shared_ptr<void>{dlopen(path, RTLD_NOW), [](void* shared_object) {
                                                   if (shared_object != nullptr) {
                                                       if (0 != dlclose(shared_object)) {
                                                           std::cerr << "dlclose failed: " << dlerror() << std::endl;
                                                       }
                                                   }
                                               }};
    if (!shared_object) {
        std::stringstream ss;
        ss << "Cannot load library '" << path << "': " << dlerror();
        throw std::runtime_error(ss.str());
    }
    return shared_object;
}

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
std::shared_ptr<void> load_shared_object(const wchar_t* path) {
    return load_shared_object(ov::util::wstring_to_string(path).c_str());
}
#endif  // OPENVINO_ENABLE_UNICODE_PATH_SUPPORT

void* get_symbol(const std::shared_ptr<void>& shared_object, const char* symbol_name) {
    if (!shared_object) {
        std::stringstream ss;
        ss << "Cannot get '" << symbol_name << "' content from unknown library!";
        throw std::runtime_error(ss.str());
    }
    void* procAddr = nullptr;
    procAddr = dlsym(shared_object.get(), symbol_name);
    if (procAddr == nullptr) {
        std::stringstream ss;
        ss << "dlSym cannot locate method '" << symbol_name << "': " << dlerror();
        throw std::runtime_error(ss.str());
    }
    return procAddr;
}

struct SharedObjectLoader::Impl {
    std::shared_ptr<void> shared_object = nullptr;

    explicit Impl(const std::shared_ptr<void>& shared_object_) : shared_object{shared_object_} {}

    explicit Impl(const char* pluginName) : shared_object{ov::util::load_shared_object(pluginName)} {}

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
    explicit Impl(const wchar_t* pluginName) : Impl(ov::util::wstring_to_string(pluginName).c_str()) {}
#endif  // OPENVINO_ENABLE_UNICODE_PATH_SUPPORT

    void* get_symbol(const char* symbolName) const {
        return ov::util::get_symbol(shared_object, symbolName);
    }
};

SharedObjectLoader::SharedObjectLoader(const std::shared_ptr<void>& shared_object) {
    _impl.reset(new Impl(shared_object));
}

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
SharedObjectLoader::SharedObjectLoader(const wchar_t* pluginName) {
    _impl.reset(new Impl(pluginName));
}
#endif

SharedObjectLoader::SharedObjectLoader(const char* pluginName) {
    _impl.reset(new Impl(pluginName));
}

SharedObjectLoader::~SharedObjectLoader() {}

void* SharedObjectLoader::get_symbol(const char* symbolName) const {
    if (_impl == nullptr) {
        std::stringstream ss;
        ss << "SharedObjectLoader is not initialized";
        throw std::runtime_error(ss.str());
    }
    return _impl->get_symbol(symbolName);
}

std::shared_ptr<void> SharedObjectLoader::get() const {
    return _impl->shared_object;
}

}  // namespace util
}  // namespace ov
