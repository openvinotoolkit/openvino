// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_extension.h"

#include "openvino/util/shared_object.hpp"

using namespace InferenceEngine;

IE_SUPPRESS_DEPRECATED_START

namespace {

template <typename T>
std::shared_ptr<T> CreateExtensionFromLibrary(std::shared_ptr<void> _so) {
    std::shared_ptr<T> _ptr = nullptr;
    constexpr char createFuncName[] = "CreateExtension";

    try {
        void* create = nullptr;
        try {
            create = ov::util::get_symbol(_so, (createFuncName + std::string("Shared")).c_str());
        } catch (const std::runtime_error&) {
        }

        if (create == nullptr) {
            create = ov::util::get_symbol(_so, createFuncName);
            using CreateF = StatusCode(T*&, ResponseDesc*);
            T* object = nullptr;
            ResponseDesc desc;
            StatusCode sts = reinterpret_cast<CreateF*>(create)(object, &desc);
            if (sts != OK) {
                IE_EXCEPTION_SWITCH(sts,
                                    ExceptionType,
                                    details::ThrowNow<ExceptionType>{} <<= std::stringstream{} << desc.msg)
            }
            IE_SUPPRESS_DEPRECATED_START
            _ptr = std::shared_ptr<T>(object, [](T* ptr) {
                ptr->Release();
            });
            IE_SUPPRESS_DEPRECATED_END
        } else {
            using CreateF = void(std::shared_ptr<T>&);
            reinterpret_cast<CreateF*>(create)(_ptr);
        }
    } catch (...) {
        details::Rethrow();
    }

    return _ptr;
}

}  // namespace

Extension::Extension(const std::string& name) {
    try {
        _so = ov::util::load_shared_object(name.c_str());
    } catch (const std::runtime_error&) {
        details::Rethrow();
    }
    _actual = CreateExtensionFromLibrary<IExtension>(_so);
}

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
Extension::Extension(const std::wstring& name) {
    try {
        _so = ov::util::load_shared_object(name.c_str());
    } catch (const std::runtime_error&) {
        details::Rethrow();
    }
    _actual = CreateExtensionFromLibrary<IExtension>(_so);
}
#endif  // OPENVINO_ENABLE_UNICODE_PATH_SUPPORT

std::map<std::string, ngraph::OpSet> Extension::getOpSets() {
    return _actual->getOpSets();
}
