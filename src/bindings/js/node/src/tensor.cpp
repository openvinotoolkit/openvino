// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tensor.hpp"

#include "addon.hpp"

TensorWrap::TensorWrap(const Napi::CallbackInfo& info) : Napi::ObjectWrap<TensorWrap>(info) {
    if (info.Length() == 0) {
        return;
    }

    if (info.Length() == 1 || info.Length() > 3) {
        reportError(info.Env(), "Invalid number of arguments for Tensor constructor.");
        return;
    }

    try {
        const auto type = js_to_cpp<ov::element::Type_t>(info, 0, {napi_string});
        const auto shape_vec = js_to_cpp<std::vector<size_t>>(info, 1, {napi_int32_array, napi_uint32_array, js_array});
        const auto& shape = ov::Shape(shape_vec);

        if (info.Length() == 2) {
            this->_tensor = ov::Tensor(type, shape);
        } else if (info.Length() == 3) {
            if (!info[2].IsTypedArray()) {
                reportError(info.Env(), "Third argument of a tensor must be of type TypedArray.");
                return;
            }

            const auto data = info[2].As<Napi::TypedArray>();
            this->_tensor = cast_to_tensor(data, shape, type);
        }
    } catch (std::invalid_argument& e) {
        reportError(info.Env(), std::string("Invalid tensor argument. ") + e.what());
    } catch (std::exception& e) {
        reportError(info.Env(), e.what());
    }
}

Napi::Function TensorWrap::get_class_constructor(Napi::Env env) {
    return DefineClass(env,
                       "TensorWrap",
                       {InstanceAccessor<&TensorWrap::get_data>("data"),
                        InstanceMethod("getData", &TensorWrap::get_data),
                        InstanceMethod("getShape", &TensorWrap::get_shape),
                        InstanceMethod("getElementType", &TensorWrap::get_element_type)});
}

Napi::Object TensorWrap::init(Napi::Env env, Napi::Object exports) {
    const auto& prototype = get_class_constructor(env);

    const auto ref = new Napi::FunctionReference();
    *ref = Napi::Persistent(prototype);
    const auto data = env.GetInstanceData<AddonData>();
    data->tensor_prototype = ref;

    exports.Set("Tensor", prototype);
    return exports;
}

ov::Tensor TensorWrap::get_tensor() const {
    return this->_tensor;
}

void TensorWrap::set_tensor(const ov::Tensor& tensor) {
    _tensor = tensor;
}

Napi::Object TensorWrap::wrap(Napi::Env env, ov::Tensor tensor) {
    const auto prototype = env.GetInstanceData<AddonData>()->tensor_prototype;
    if (!prototype) {
        OPENVINO_THROW("Invalid pointer to Tensor prototype.");
    }
    auto tensor_js = prototype->New({});
    const auto t = Napi::ObjectWrap<TensorWrap>::Unwrap(tensor_js);
    t->set_tensor(tensor);
    return tensor_js;
}

Napi::Value TensorWrap::get_data(const Napi::CallbackInfo& info) {
    auto type = _tensor.get_element_type();

    switch (type) {
    case ov::element::Type_t::i8: {
        auto arr = Napi::Int8Array::New(info.Env(), _tensor.get_size());
        std::memcpy(arr.Data(), _tensor.data(), _tensor.get_byte_size());
        return arr;
    }
    case ov::element::Type_t::u8: {
        auto arr = Napi::Uint8Array::New(info.Env(), _tensor.get_size());
        std::memcpy(arr.Data(), _tensor.data(), _tensor.get_byte_size());
        return arr;
    }
    case ov::element::Type_t::i16: {
        auto arr = Napi::Int16Array::New(info.Env(), _tensor.get_size());
        std::memcpy(arr.Data(), _tensor.data(), _tensor.get_byte_size());
        return arr;
    }
    case ov::element::Type_t::u16: {
        auto arr = Napi::Uint16Array::New(info.Env(), _tensor.get_size());
        std::memcpy(arr.Data(), _tensor.data(), _tensor.get_byte_size());
        return arr;
    }
    case ov::element::Type_t::i32: {
        auto arr = Napi::Int32Array::New(info.Env(), _tensor.get_size());
        std::memcpy(arr.Data(), _tensor.data(), _tensor.get_byte_size());
        return arr;
    }
    case ov::element::Type_t::u32: {
        auto arr = Napi::Uint32Array::New(info.Env(), _tensor.get_size());
        std::memcpy(arr.Data(), _tensor.data(), _tensor.get_byte_size());
        return arr;
    }
    case ov::element::Type_t::f32: {
        auto arr = Napi::Float32Array::New(info.Env(), _tensor.get_size());
        std::memcpy(arr.Data(), _tensor.data(), _tensor.get_byte_size());
        return arr;
    }
    case ov::element::Type_t::f64: {
        auto arr = Napi::Float64Array::New(info.Env(), _tensor.get_size());
        std::memcpy(arr.Data(), _tensor.data(), _tensor.get_byte_size());
        return arr;
    }
    case ov::element::Type_t::i64: {
        auto arr = Napi::BigInt64Array::New(info.Env(), _tensor.get_size());
        std::memcpy(arr.Data(), _tensor.data(), _tensor.get_byte_size());
        return arr;
    }
    case ov::element::Type_t::u64: {
        auto arr = Napi::BigUint64Array::New(info.Env(), _tensor.get_size());
        std::memcpy(arr.Data(), _tensor.data(), _tensor.get_byte_size());
        return arr;
    }
    default: {
        reportError(info.Env(), "Failed to return tensor data.");
        return info.Env().Null();
    }
    }
}

Napi::Value TensorWrap::get_shape(const Napi::CallbackInfo& info) {
    return cpp_to_js<ov::Shape, Napi::Array>(info, _tensor.get_shape());
}

Napi::Value TensorWrap::get_element_type(const Napi::CallbackInfo& info) {
    return cpp_to_js<ov::element::Type_t, Napi::String>(info, _tensor.get_element_type());
}
