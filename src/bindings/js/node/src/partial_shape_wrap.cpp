// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "partial_shape_wrap.hpp"

#include "addon.hpp"

PartialShapeWrap::PartialShapeWrap(const Napi::CallbackInfo& info) : Napi::ObjectWrap<PartialShapeWrap>(info) {
    const size_t attrs_length = info.Length();

    if (attrs_length == 1 && info[0].IsString()) {
        try {
            const auto& shape = std::string(info[0].ToString());

            _partial_shape = ov::PartialShape(shape);
        } catch (std::exception& e) {
            reportError(info.Env(), e.what());
        }
    } else {
        reportError(info.Env(), "Invalid parameters for PartialShape constructor.");
    }
}

Napi::Function PartialShapeWrap::get_class_constructor(Napi::Env env) {
    return DefineClass(env,
                       "PartialShapeWrap",
                       {
                           InstanceMethod("isStatic", &PartialShapeWrap::is_static),
                           InstanceMethod("isDynamic", &PartialShapeWrap::is_dynamic),
                           InstanceMethod("toString", &PartialShapeWrap::to_string),
                           InstanceMethod("getDimensions", &PartialShapeWrap::get_dimensions),
                       });
}

Napi::Object PartialShapeWrap::init(Napi::Env env, Napi::Object exports) {
    const auto& prototype = get_class_constructor(env);

    const auto ref = new Napi::FunctionReference();
    *ref = Napi::Persistent(prototype);
    const auto data = env.GetInstanceData<AddonData>();
    data->partial_shape_prototype = ref;

    exports.Set("PartialShape", prototype);
    return exports;
}

Napi::Object PartialShapeWrap::wrap(Napi::Env env, ov::PartialShape partial_shape) {
    const auto prototype = env.GetInstanceData<AddonData>()->partial_shape_prototype;
    if (!prototype) {
        OPENVINO_THROW("Invalid pointer to PartialShape prototype.");
    }
    auto obj = prototype->New({});
    const auto t = Napi::ObjectWrap<PartialShapeWrap>::Unwrap(obj);
    t->_partial_shape = partial_shape;

    return obj;
}

Napi::Value PartialShapeWrap::is_static(const Napi::CallbackInfo& info) {
    return cpp_to_js<bool, Napi::Boolean>(info, _partial_shape.is_static());
}

Napi::Value PartialShapeWrap::is_dynamic(const Napi::CallbackInfo& info) {
    return cpp_to_js<bool, Napi::Boolean>(info, _partial_shape.is_dynamic());
}

Napi::Value PartialShapeWrap::to_string(const Napi::CallbackInfo& info) {
    return Napi::String::New(info.Env(), _partial_shape.to_string());
}

Napi::Value PartialShapeWrap::get_dimensions(const Napi::CallbackInfo& info) {
    return cpp_to_js<ov::PartialShape, Napi::Array>(info, _partial_shape);
}
