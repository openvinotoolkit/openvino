// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "node/include/element_type.hpp"

#include "node/include/helper.hpp"

namespace element {
Napi::Object init(Napi::Env env, Napi::Object exports) {
    auto element = Napi::PropertyDescriptor::Accessor<add_element_namespace>("element");

    exports.DefineProperty(element);

    return exports;
}

Napi::Value add_element_namespace(const Napi::CallbackInfo& info) {
    auto element = Napi::Object::New(info.Env());
    std::vector<Napi::PropertyDescriptor> pds;

    for (const auto& et : get_supported_types())
        pds.push_back(Napi::PropertyDescriptor::Value(et, Napi::String::New(info.Env(), et), napi_default));

    element.DefineProperties(pds);

    return element;
}
};  // namespace element
