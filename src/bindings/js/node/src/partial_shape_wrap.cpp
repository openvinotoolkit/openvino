// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "node/include/partial_shape_wrap.hpp"

#include "node/include/addon.hpp"
#include "node/include/errors.hpp"
#include "node/include/helper.hpp"
#include "node/include/type_validation.hpp"

PartialShapeWrap::PartialShapeWrap(const Napi::CallbackInfo& info) : Napi::ObjectWrap<PartialShapeWrap>(info) {
    std::vector<std::string> allowed_signatures;

    try {
        if (ov::js::validate<Napi::String>(info, allowed_signatures)) {
            _partial_shape = ov::PartialShape(info[0].ToString());
        } else if (ov::js::validate(info, allowed_signatures)) {
            return;
        } else {
            OPENVINO_THROW("'PartialShape' constructor", ov::js::get_parameters_error_msg(info, allowed_signatures));
        }
    } catch (std::exception& err) {
        reportError(info.Env(), err.what());
    }
}

Napi::Function PartialShapeWrap::get_class(Napi::Env env) {
    return DefineClass(env,
                       "PartialShapeWrap",
                       {
                           InstanceMethod("isStatic", &PartialShapeWrap::is_static),
                           InstanceMethod("isDynamic", &PartialShapeWrap::is_dynamic),
                           InstanceMethod("toString", &PartialShapeWrap::to_string),
                           InstanceMethod("getDimensions", &PartialShapeWrap::get_dimensions),
                       });
}

Napi::Object PartialShapeWrap::wrap(Napi::Env env, ov::PartialShape partial_shape) {
    const auto& prototype = env.GetInstanceData<AddonData>()->partial_shape;
    if (!prototype) {
        OPENVINO_THROW("Invalid pointer to PartialShape prototype.");
    }
    auto obj = prototype.New({});
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
