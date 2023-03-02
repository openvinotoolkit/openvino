#include "helper.hpp"

#include <iostream>

napi_types napiType(Napi::Value val) {
    if (val.IsTypedArray())
        return val.As<Napi::TypedArray>().TypedArrayType();
    else if (val.IsArray())
        return js_array;
    else
        return val.Type();
}

bool acceptableType(Napi::Value val, const std::vector<napi_types>& acceptable) {
    return std::any_of(acceptable.begin(), acceptable.end(), [val](napi_types t) {
        return napiType(val) == t;
    });
}

template <>
int32_t js_to_cpp<int32_t>(const Napi::CallbackInfo& info,
                           const size_t idx,
                           const std::vector<napi_types>& acceptable_types) {
    const auto elem = info[idx];
    if (!acceptableType(elem, acceptable_types))
        throw std::invalid_argument(std::string("Cannot convert argument" + std::to_string(idx)));
    if (!elem.IsNumber()) {
        throw std::invalid_argument(std::string("Passed argument must be a number."));
    }
    return elem.ToNumber().Int32Value();
}

template <>
std::string js_to_cpp<std::string>(const Napi::CallbackInfo& info,
                                   const size_t idx,
                                   const std::vector<napi_types>& acceptable_types) {
    const auto elem = info[idx];
    if (!acceptableType(elem, acceptable_types))
        throw std::invalid_argument(std::string("Cannot convert argument") + std::to_string(idx));
    if (!elem.IsString()) {
        throw std::invalid_argument(std::string("Passed argument must be a string."));
    }
    return elem.ToString();
}

template <>
std::vector<size_t> js_to_cpp<std::vector<size_t>>(const Napi::CallbackInfo& info,
                                                   const size_t idx,
                                                   const std::vector<napi_types>& acceptable_types) {
    const auto elem = info[idx];
    if (!acceptableType(elem, acceptable_types))
        throw std::invalid_argument(std::string("Cannot convert argument") + std::to_string(idx));
    if (!elem.IsArray() && !elem.IsTypedArray()) {
        throw std::invalid_argument(std::string("Passed argument must be of type Array or TypedArray."));
    } else if (elem.IsArray()) {
        auto array = elem.As<Napi::Array>();
        size_t arrayLength = array.Length();

        std::vector<size_t> nativeArray;

        for (size_t i = 0; i < arrayLength; i++) {
            Napi::Value arrayItem = array[i];
            if (!arrayItem.IsNumber()) {
                throw std::invalid_argument(std::string("Passed array must contain only numbers."));
            }
            Napi::Number num = arrayItem.As<Napi::Number>();
            nativeArray.push_back(static_cast<size_t>(num.Int32Value()));
        }
        return nativeArray;

    } else {  //( elem.IsTypedArray()){

        if ((elem.As<Napi::TypedArray>().TypedArrayType() != 5)) {
            throw std::invalid_argument(std::string("Passed argument must be a Int32Array."));
        }
        auto buf = elem.As<Napi::Int32Array>();
        auto data_ptr = static_cast<int*>(buf.ArrayBuffer().Data());
        std::vector<size_t> vector(data_ptr, data_ptr + 4);
        return vector;
    }
}

template <>
ov::element::Type_t js_to_cpp<ov::element::Type_t>(const Napi::CallbackInfo& info,
                                                   const size_t idx,
                                                   const std::vector<napi_types>& acceptable_types) {
    const auto elem = info[idx];
    if (!acceptableType(elem, acceptable_types))
        throw std::invalid_argument(std::string("Cannot convert Napi::Value to ov::element::Type_t"));
    const auto key = elem.ToString();
    if (element_type_map.find(key) == element_type_map.end())
        throw std::invalid_argument(std::string("Cannot create ov element Type"));

    return element_type_map.at(key);
}

template <>
ov::Layout js_to_cpp<ov::Layout>(const Napi::CallbackInfo& info,
                                 const size_t idx,
                                 const std::vector<napi_types>& acceptable_types) {
    auto layout = js_to_cpp<std::string>(info, idx, acceptable_types);
    return ov::Layout(layout);
}

template <>
ov::Shape js_to_cpp<ov::Shape>(const Napi::CallbackInfo& info,
                               const size_t idx,
                               const std::vector<napi_types>& acceptable_types) {
    auto shape = js_to_cpp<std::vector<size_t>>(info, idx, acceptable_types);
    return ov::Shape(shape);
}

template <>
Napi::String cpp_to_js<ov::element::Type_t, Napi::String>(const Napi::CallbackInfo& info, ov::element::Type_t type) {
    auto str = Napi::String::New(info.Env(), "");
    for (auto& it : element_type_map)
        if (it.second == type) {
            str = Napi::String::New(info.Env(), it.first);
            break;
        }
    return str;
}
