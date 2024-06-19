// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "node/include/tensor.hpp"

#include "node/include/addon.hpp"
#include "node/include/errors.hpp"
#include "node/include/helper.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"

TensorWrap::TensorWrap(const Napi::CallbackInfo& info) : Napi::ObjectWrap<TensorWrap>(info) {
    if (info.Length() == 0) {
        return;
    }
    try {
        OPENVINO_ASSERT(info.Length() <= 3, "Invalid number of arguments for Tensor constructor.");
        if (info.Length() == 1 && info[0].IsArray()) {
            this->_tensor = cast_to_tensor(info[0].As<Napi::Array>());
        } else {
            const auto type = js_to_cpp<ov::element::Type_t>(info, 0);
            const auto& shape = js_to_cpp<ov::Shape>(info, 1);

            if (info.Length() == 2) {
                this->_tensor = ov::Tensor(type, shape);
            } else if (info.Length() == 3 && info[2].IsTypedArray()) {
                this->_tensor = cast_to_tensor(info[2].As<Napi::TypedArray>(), shape, type);
            } else {
                OPENVINO_THROW("Third argument of a tensor must be TypedArray.");
            }
        }
    } catch (std::invalid_argument& e) {
        reportError(info.Env(), std::string("Invalid tensor argument. ") + e.what());
    } catch (std::exception& e) {
        reportError(info.Env(), e.what());
    }
}

Napi::Function TensorWrap::get_class(Napi::Env env) {
    return DefineClass(env,
                       "TensorWrap",
                       {InstanceAccessor<&TensorWrap::get_data, &TensorWrap::set_data>("data"),
                        InstanceMethod("getData", &TensorWrap::get_data),
                        InstanceMethod("getShape", &TensorWrap::get_shape),
                        InstanceMethod("getElementType", &TensorWrap::get_element_type),
                        InstanceMethod("getSize", &TensorWrap::get_size)});
}

ov::Tensor TensorWrap::get_tensor() const {
    return this->_tensor;
}

void TensorWrap::set_tensor(const ov::Tensor& tensor) {
    _tensor = tensor;
}

Napi::Object TensorWrap::wrap(Napi::Env env, ov::Tensor tensor) {
    const auto& prototype = env.GetInstanceData<AddonData>()->tensor;
    if (!prototype) {
        OPENVINO_THROW("Invalid pointer to Tensor prototype.");
    }
    auto tensor_js = prototype.New({});
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
    case ov::element::string: {
        auto str_data = Napi::Array::New(info.Env(), _tensor.get_size());
        const auto data = _tensor.data<std::string>();
        for (uint32_t i = 0; i < _tensor.get_size(); ++i) {
            str_data[i] = Napi::String::New(info.Env(), data[i]);
        }
        return str_data;
    }
    default: {
        reportError(info.Env(), "Failed to return tensor data.");
        return info.Env().Null();
    }
    }
}

void TensorWrap::set_data(const Napi::CallbackInfo& info, const Napi::Value& value) {
    try {
        if (value.IsTypedArray()) {
            const auto buf = value.As<Napi::TypedArray>();

            if (_tensor.get_byte_size() != buf.ByteLength()) {
                OPENVINO_THROW("Passed array must have the same size as the Tensor!");
            }
            const auto napi_type = buf.TypedArrayType();
            std::memcpy(_tensor.data(get_ov_type(napi_type)), buf.ArrayBuffer().Data(), _tensor.get_byte_size());
        } else if (value.IsArray()) {
            fill_tensor_from_strings(_tensor, value.As<Napi::Array>());
        } else {
            OPENVINO_THROW("Passed argument must be TypedArray, or Array if the tensor type is string.");
        }
    } catch (std::exception& e) {
        reportError(info.Env(), e.what());
    }
}

Napi::Value TensorWrap::get_shape(const Napi::CallbackInfo& info) {
    if (info.Length() > 0) {
        reportError(info.Env(), "No parameters are allowed for the getShape() method.");
        return info.Env().Undefined();
    }
    return cpp_to_js<ov::Shape, Napi::Array>(info, _tensor.get_shape());
}

Napi::Value TensorWrap::get_element_type(const Napi::CallbackInfo& info) {
    return cpp_to_js<ov::element::Type_t, Napi::String>(info, _tensor.get_element_type());
}

Napi::Value TensorWrap::get_size(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() > 0) {
        reportError(env, "getSize() does not accept any arguments.");
        return env.Undefined();
    }
    const auto size = static_cast<double>(_tensor.get_size());
    return Napi::Number::New(env, size);
}
