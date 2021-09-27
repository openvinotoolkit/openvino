// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "details/ie_so_loader.h"
#include "file_utils.h"
#include "ie_common.h"
#include "openvino/util/so_loader.hpp"

//
// LoadLibraryA, LoadLibraryW:
//  WINAPI_FAMILY_DESKTOP_APP - OK (default)
//  WINAPI_FAMILY_PC_APP - FAIL ?? (defined by cmake)
//  WINAPI_FAMILY_PHONE_APP - FAIL ??
//  WINAPI_FAMILY_GAMES - OK
//  WINAPI_FAMILY_SERVER - OK
//  WINAPI_FAMILY_SYSTEM - OK
//
// GetModuleHandleExA, GetModuleHandleExW:
//  WINAPI_FAMILY_DESKTOP_APP - OK (default)
//  WINAPI_FAMILY_PC_APP - FAIL ?? (defined by cmake)
//  WINAPI_FAMILY_PHONE_APP - FAIL ??
//  WINAPI_FAMILY_GAMES - OK
//  WINAPI_FAMILY_SERVER - OK
//  WINAPI_FAMILY_SYSTEM - OK
//
// GetModuleHandleA, GetModuleHandleW:
//  WINAPI_FAMILY_DESKTOP_APP - OK (default)
//  WINAPI_FAMILY_PC_APP - FAIL ?? (defined by cmake)
//  WINAPI_FAMILY_PHONE_APP - FAIL ??
//  WINAPI_FAMILY_GAMES - OK
//  WINAPI_FAMILY_SERVER - OK
//  WINAPI_FAMILY_SYSTEM - OK
//
// SetDllDirectoryA, SetDllDirectoryW:
//  WINAPI_FAMILY_DESKTOP_APP - OK (default)
//  WINAPI_FAMILY_PC_APP - FAIL ?? (defined by cmake)
//  WINAPI_FAMILY_PHONE_APP - FAIL ??
//  WINAPI_FAMILY_GAMES - OK
//  WINAPI_FAMILY_SERVER - FAIL
//  WINAPI_FAMILY_SYSTEM - FAIL
//
// GetDllDirectoryA, GetDllDirectoryW:
//  WINAPI_FAMILY_DESKTOP_APP - FAIL
//  WINAPI_FAMILY_PC_APP - FAIL (defined by cmake)
//  WINAPI_FAMILY_PHONE_APP - FAIL
//  WINAPI_FAMILY_GAMES - FAIL
//  WINAPI_FAMILY_SERVER - FAIL
//  WINAPI_FAMILY_SYSTEM - FAIL
//
// SetupDiGetClassDevsA, SetupDiEnumDeviceInfo, SetupDiGetDeviceInstanceIdA, SetupDiDestroyDeviceInfoList:
//  WINAPI_FAMILY_DESKTOP_APP - FAIL (default)
//  WINAPI_FAMILY_PC_APP - FAIL (defined by cmake)
//  WINAPI_FAMILY_PHONE_APP - FAIL
//  WINAPI_FAMILY_GAMES - FAIL
//  WINAPI_FAMILY_SERVER - FAIL
//  WINAPI_FAMILY_SYSTEM - FAIL
//

#if defined(WINAPI_FAMILY) && !WINAPI_PARTITION_DESKTOP
#    error "Only WINAPI_PARTITION_DESKTOP is supported, because of LoadLibrary[A|W]"
#endif

#include <direct.h>

#include <mutex>

#ifndef NOMINMAX
#    define NOMINMAX
#endif

#include <windows.h>

#include "openvino/util/so_loader.hpp"

namespace InferenceEngine {
namespace details {
struct SharedObjectLoader::Impl : public ov::util::SharedObjectLoader {
    explicit Impl(const std::shared_ptr<void>& shared_object_) : SharedObjectLoader(shared_object_) {}

    explicit Impl(const char* pluginName) : SharedObjectLoader(pluginName) {}

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
    explicit Impl(const wchar_t* pluginName) : SharedObjectLoader(pluginName) {}
#endif  // OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
};

SharedObjectLoader::SharedObjectLoader(const std::shared_ptr<void>& shared_object) {
    _impl.reset(new Impl(shared_object));
}

SharedObjectLoader::~SharedObjectLoader() {}

SharedObjectLoader::SharedObjectLoader(const char* pluginName) {
    _impl = std::make_shared<Impl>(pluginName);
}
#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
SharedObjectLoader::SharedObjectLoader(const wchar_t* pluginName) {
    _impl = std::make_shared<Impl>(pluginName);
}
#endif

void* SharedObjectLoader::get_symbol(const char* symbolName) const {
    if (_impl == nullptr) {
        IE_THROW(NotAllocated) << "SharedObjectLoader is not initialized";
    }
    return _impl->get_symbol(symbolName);
}

std::shared_ptr<void> SharedObjectLoader::get() const {
    return _impl->get();
}

}  // namespace details
}  // namespace InferenceEngine
