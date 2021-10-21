// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "details/ie_so_loader.h"
#include "ie_common.h"
#include "openvino/util/file_util.hpp"
#include "openvino/util/shared_object.hpp"

namespace InferenceEngine {
namespace details {

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
        IE_THROW(NotAllocated) << "SharedObjectLoader is not initialized";
    }
    return _impl->get_symbol(symbolName);
}

std::shared_ptr<void> SharedObjectLoader::get() const {
    return _impl->shared_object;
}

}  // namespace details
}  // namespace InferenceEngine
