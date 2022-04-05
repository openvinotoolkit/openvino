// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "details/ie_so_loader.h"
#include "ie_common.h"
#include "openvino/util/file_util.hpp"
#include "openvino/util/shared_object.hpp"

IE_SUPPRESS_DEPRECATED_START

namespace InferenceEngine {
namespace details {

SharedObjectLoader::SharedObjectLoader(const std::shared_ptr<void>& so) : _so(so) {}

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
SharedObjectLoader::SharedObjectLoader(const wchar_t* pluginName)
    : SharedObjectLoader(ov::util::wstring_to_string(pluginName).c_str()) {}
#endif

SharedObjectLoader::SharedObjectLoader(const char* pluginName) : _so{nullptr} {
    try {
        _so = ov::util::load_shared_object(pluginName);
    } catch (const std::runtime_error& ex) {
        IE_THROW(GeneralError) << ex.what();
    }
}

SharedObjectLoader::~SharedObjectLoader() {}

void* SharedObjectLoader::get_symbol(const char* symbolName) const {
    try {
        return ov::util::get_symbol(_so, symbolName);
    } catch (const std::runtime_error& ex) {
        IE_THROW(NotFound) << ex.what();
    }
}

std::shared_ptr<void> SharedObjectLoader::get() const {
    return _so;
}

}  // namespace details
}  // namespace InferenceEngine

IE_SUPPRESS_DEPRECATED_END
