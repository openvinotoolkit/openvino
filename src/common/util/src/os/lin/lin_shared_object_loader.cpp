// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <dlfcn.h>

#include <iostream>
#include <sstream>

#include "openvino/util/file_util.hpp"
#include "openvino/util/shared_object.hpp"

namespace ov {
namespace util {
std::shared_ptr<void> load_shared_object(const char* path) {
    auto shared_object = std::shared_ptr<void>{dlopen(path, RTLD_NOW), [](void* shared_object) {
                                                   if (shared_object != nullptr) {
                                                       if (0 != dlclose(shared_object)) {
                                                           std::cerr << "dlclose failed";
                                                           if (auto error = dlerror()) {
                                                               std::cerr << ": " << error;
                                                           }
                                                           std::cerr << std::endl;
                                                       }
                                                   }
                                               }};
    if (!shared_object) {
        std::stringstream ss;
        ss << "Cannot load library '" << path << "'";
        if (auto error = dlerror()) {
            ss << ": " << error;
        }
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
}  // namespace util
}  // namespace ov
