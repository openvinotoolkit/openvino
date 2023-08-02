// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "element_type.hpp"

#include <iostream>
#include <typeinfo>

Napi::Value enumElementType(const Napi::CallbackInfo& info) {
    Napi::Object enumObj = Napi::Object::New(info.Env());
    std::vector<Napi::PropertyDescriptor> pds;

    for (const auto& et : get_supported_types())
        pds.push_back(Napi::PropertyDescriptor::Value(et, Napi::String::New(info.Env(), et), napi_default));

    enumObj.DefineProperties(pds);
    return enumObj;
}
