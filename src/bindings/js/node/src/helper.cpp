// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "node/include/helper.hpp"

#include "node/include/compiled_model.hpp"
#include "node/include/tensor.hpp"
#include "node/include/type_validation.hpp"

const std::vector<std::string>& get_supported_types() {
    static const std::vector<std::string> supported_element_types =
        {"i8", "u8", "i16", "u16", "i32", "u32", "f32", "f64", "i64", "u64", "string"};
    return supported_element_types;
}

const ov::element::Type_t& get_ov_type(napi_typedarray_type type) {
    static const std::unordered_map<napi_typedarray_type, ov::element::Type_t> typedarray_to_ov_type{
        {napi_int8_array, ov::element::Type_t::i8},
        {napi_uint8_array, ov::element::Type_t::u8},
        {napi_int16_array, ov::element::Type_t::i16},
        {napi_uint16_array, ov::element::Type_t::u16},
        {napi_int32_array, ov::element::Type_t::i32},
        {napi_uint32_array, ov::element::Type_t::u32},
        {napi_float32_array, ov::element::Type_t::f32},
        {napi_float64_array, ov::element::Type_t::f64},
        {napi_bigint64_array, ov::element::Type_t::i64},
        {napi_biguint64_array, ov::element::Type_t::u64}};
    return typedarray_to_ov_type.at(type);
}

napi_types napiType(const Napi::Value& val) {
    if (val.IsTypedArray())
        return val.As<Napi::TypedArray>().TypedArrayType();
    else if (val.IsArray())
        return js_array;
    else
        return val.Type();
}

template <>
std::string js_to_cpp<std::string>(const Napi::CallbackInfo& info, const size_t idx) {
    const auto elem = info[idx];
    if (!elem.IsString()) {
        OPENVINO_THROW("Passed argument must be a string.");
    }
    return elem.ToString();
}

template <>
std::vector<size_t> js_to_cpp<std::vector<size_t>>(const Napi::CallbackInfo& info, const size_t idx) {
    const auto elem = info[idx];
    if (elem.IsArray()) {
        auto array = elem.As<Napi::Array>();
        size_t arrayLength = array.Length();

        std::vector<size_t> nativeArray;

        for (uint32_t i = 0; i < arrayLength; ++i) {
            Napi::Value arrayItem = array[i];
            if (!arrayItem.IsNumber()) {
                OPENVINO_THROW(std::string("Passed array must contain only numbers."));
            }
            Napi::Number num = arrayItem.As<Napi::Number>();
            nativeArray.push_back(static_cast<size_t>(num.Int32Value()));
        }
        return nativeArray;

    } else if (elem.IsTypedArray()) {
        Napi::TypedArray buf;
        napi_typedarray_type type = elem.As<Napi::TypedArray>().TypedArrayType();
        if ((type != napi_int32_array) && (type != napi_uint32_array)) {
            OPENVINO_THROW("Passed argument must be an Int32Array or a Uint32Array.");
        } else if (type == napi_uint32_array)
            buf = elem.As<Napi::Uint32Array>();
        else {
            buf = elem.As<Napi::Int32Array>();
        }
        auto data_ptr = static_cast<int*>(buf.ArrayBuffer().Data());
        std::vector<size_t> vector(data_ptr, data_ptr + buf.ElementLength());
        return vector;
    } else {
        OPENVINO_THROW("Passed argument must be of type Array or TypedArray.");
    }
}

template <>
std::unordered_set<std::string> js_to_cpp<std::unordered_set<std::string>>(const Napi::CallbackInfo& info,
                                                                           const size_t idx) {
    const auto elem = info[idx];
    if (!elem.IsArray()) {
        OPENVINO_THROW(std::string("Passed argument must be of type Array."));
    } else {
        auto array = elem.As<Napi::Array>();
        size_t arrayLength = array.Length();

        std::unordered_set<std::string> nativeArray;

        for (uint32_t i = 0; i < arrayLength; ++i) {
            Napi::Value arrayItem = array[i];
            if (!arrayItem.IsString()) {
                OPENVINO_THROW(std::string("Passed array must contain only strings."));
            }
            Napi::String str = arrayItem.As<Napi::String>();
            nativeArray.insert(str.Utf8Value());
        }
        return nativeArray;
    }
}

template <>
ov::element::Type_t js_to_cpp<ov::element::Type_t>(const Napi::CallbackInfo& info, const size_t idx) {
    const auto elem = info[idx];
    OPENVINO_ASSERT(elem.IsString(), "Passed argument must be of type String.");

    const std::string type = elem.ToString();
    const auto& types = get_supported_types();
    if (std::find(types.begin(), types.end(), type) == types.end())
        OPENVINO_THROW("Cannot create ov::element::Type");

    return static_cast<ov::element::Type_t>(ov::element::Type(type));
}

template <>
ov::Layout js_to_cpp<ov::Layout>(const Napi::CallbackInfo& info, const size_t idx) {
    const auto& layout = js_to_cpp<std::string>(info, idx);
    return ov::Layout(layout);
}

template <>
ov::Shape js_to_cpp<ov::Shape>(const Napi::CallbackInfo& info, const size_t idx) {
    const auto& shape = js_to_cpp<std::vector<size_t>>(info, idx);
    return ov::Shape(shape);
}

template <>
ov::preprocess::ResizeAlgorithm js_to_cpp<ov::preprocess::ResizeAlgorithm>(const Napi::CallbackInfo& info,
                                                                           const size_t idx) {
    const auto& elem = info[idx];
    OPENVINO_ASSERT(elem.IsString(), "Cannot convert Napi::Value to resizeAlgorithm");

    const std::string& algorithm = elem.ToString();
    if (algorithm == "RESIZE_CUBIC") {
        return ov::preprocess::ResizeAlgorithm::RESIZE_CUBIC;
    } else if (algorithm == "RESIZE_NEAREST") {
        return ov::preprocess::ResizeAlgorithm::RESIZE_NEAREST;
    } else if (algorithm == "RESIZE_LINEAR") {
        return ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR;
    } else {
        OPENVINO_THROW("Not supported resizeAlgorithm.");
    }
}

template <>
ov::Any js_to_cpp<ov::Any>(const Napi::Env& env, const Napi::Value& value) {
    if (value.IsString()) {
        return ov::Any(value.ToString().Utf8Value());
    } else if (value.IsBigInt()) {
        Napi::BigInt big_value = value.As<Napi::BigInt>();
        bool is_lossless;
        int64_t big_num = big_value.Int64Value(&is_lossless);

        if (!is_lossless) {
            OPENVINO_THROW("Result of BigInt conversion to int64_t results in a loss of precision");
        }

        return ov::Any(big_num);
    } else if (value.IsNumber()) {
        Napi::Number num = value.ToNumber();

        if (is_napi_value_int(env, value)) {
            return ov::Any(num.Int32Value());
        } else {
            return ov::Any(num.DoubleValue());
        }
    } else if (value.IsBoolean()) {
        return ov::Any(value.ToBoolean());
    } else {
        OPENVINO_THROW("Cannot convert to ov::Any");
    }
}

template <>
std::map<std::string, ov::Any> js_to_cpp<std::map<std::string, ov::Any>>(const Napi::CallbackInfo& info,
                                                                         const size_t idx) {
    const auto elem = info[idx];
    OPENVINO_ASSERT(elem.IsObject(),
                    static_cast<std::string>("Argument #" + std::to_string(idx) + " must be an Object."));

    std::map<std::string, ov::Any> properties_to_cpp;
    const auto& config = elem.ToObject();
    const auto& keys = config.GetPropertyNames();

    for (uint32_t i = 0; i < keys.Length(); ++i) {
        const std::string& option = static_cast<Napi::Value>(keys[i]).ToString();
        properties_to_cpp[option] = js_to_cpp<ov::Any>(info.Env(), config.Get(option));
    }

    return properties_to_cpp;
}

template <>
Napi::String cpp_to_js<ov::element::Type_t, Napi::String>(const Napi::CallbackInfo& info,
                                                          const ov::element::Type_t type) {
    return Napi::String::New(info.Env(), ov::element::Type(type).to_string());
}

template <>
Napi::Array cpp_to_js<ov::Shape, Napi::Array>(const Napi::CallbackInfo& info, const ov::Shape shape) {
    auto arr = Napi::Array::New(info.Env(), shape.size());
    for (uint32_t i = 0; i < shape.size(); ++i)
        arr[i] = shape[i];
    return arr;
}

template <>
Napi::Array cpp_to_js<ov::PartialShape, Napi::Array>(const Napi::CallbackInfo& info, const ov::PartialShape shape) {
    size_t size = shape.size();
    Napi::Array dimensions = Napi::Array::New(info.Env(), size);

    for (uint32_t i = 0; i < size; i++) {
        ov::Dimension dim = shape[i];

        if (dim.is_static()) {
            dimensions[i] = dim.get_length();
            continue;
        }

        auto min = dim.get_min_length();
        auto max = dim.get_max_length();

        if (min > max) {
            dimensions[i] = -1;
            continue;
        }

        dimensions[i] = cpp_to_js<ov::Dimension, Napi::Array>(info, dim);
    }

    return dimensions;
}

template <>
Napi::Array cpp_to_js<ov::Dimension, Napi::Array>(const Napi::CallbackInfo& info, const ov::Dimension dim) {
    Napi::Array interval = Napi::Array::New(info.Env(), 2);

    // Indexes looks wierd, but clear assignment,
    // like: interval[0] = value doesn't work here
    uint32_t indexes[] = {0, 1};
    interval[indexes[0]] = dim.get_min_length();
    interval[indexes[1]] = dim.get_max_length();

    return interval;
}

Napi::Object cpp_to_js(const Napi::Env& env, std::shared_ptr<ov::Model> model) {
    const auto& prototype = env.GetInstanceData<AddonData>()->model;
    if (!prototype) {
        OPENVINO_THROW("Invalid pointer to Model prototype.");
    }
    const auto& model_js = prototype.New({});
    const auto mw = Napi::ObjectWrap<ModelWrap>::Unwrap(model_js);
    mw->set_model(model);
    return model_js;
}

template <>
Napi::Boolean cpp_to_js<bool, Napi::Boolean>(const Napi::CallbackInfo& info, const bool value) {
    return Napi::Boolean::New(info.Env(), value);
}

Napi::Object cpp_to_js(const Napi::Env& env, const ov::CompiledModel& compiled_model) {
    const auto& prototype = env.GetInstanceData<AddonData>()->compiled_model;
    if (!prototype) {
        OPENVINO_THROW("Invalid pointer to CompiledModel prototype.");
    }
    auto obj = prototype.New({});
    const auto cm = Napi::ObjectWrap<CompiledModelWrap>::Unwrap(obj);
    cm->set_compiled_model(compiled_model);
    return obj;
}

ov::TensorVector parse_input_data(const Napi::Value& input) {
    ov::TensorVector parsed_input;
    if (input.IsArray()) {
        auto inputs = input.As<Napi::Array>();
        for (uint32_t i = 0; i < inputs.Length(); ++i) {
            parsed_input.emplace_back(cast_to_tensor(static_cast<Napi::Value>(inputs[i])));
        }
    } else if (input.IsObject()) {
        auto inputs = input.ToObject();
        const auto& keys = inputs.GetPropertyNames();
        for (uint32_t i = 0; i < keys.Length(); ++i) {
            auto value = inputs.Get(static_cast<Napi::Value>(keys[i]).ToString().Utf8Value());
            parsed_input.emplace_back(cast_to_tensor(static_cast<Napi::Value>(value)));
        }
    } else {
        OPENVINO_THROW("parse_input_data(): wrong arg");
    }
    return parsed_input;
}

ov::Tensor get_request_tensor(ov::InferRequest& infer_request, const std::string key) {
    return infer_request.get_tensor(key);
}

ov::Tensor get_request_tensor(ov::InferRequest& infer_request, const size_t idx) {
    return infer_request.get_input_tensor(idx);
}

ov::Tensor cast_to_tensor(const Napi::Value& value) {
    if (value.IsObject()) {
        auto tensor_wrap = Napi::ObjectWrap<TensorWrap>::Unwrap(value.ToObject());
        return tensor_wrap->get_tensor();
    } else {
        OPENVINO_THROW("Cannot create a tensor from the passed Napi::Value.");
    }
}

ov::Tensor cast_to_tensor(const Napi::CallbackInfo& info, int index) {
    if (!ov::js::validate_value<TensorWrap>(info.Env(), info[index])) {
        OPENVINO_THROW(std::string("Argument #" + std::to_string(index) + " must be a Tensor."));
    }
    const auto tensor_wrap = Napi::ObjectWrap<TensorWrap>::Unwrap(info[index].ToObject());
    return tensor_wrap->get_tensor();
}

ov::Tensor cast_to_tensor(const Napi::TypedArray& typed_array,
                          const ov::Shape& shape,
                          const ov::element::Type_t& type) {
    /* The difference between TypedArray::ArrayBuffer::Data() and e.g. Float32Array::Data() is byteOffset
    because the TypedArray may have a non-zero `ByteOffset()` into the `ArrayBuffer`. */
    if (typed_array.ByteOffset() != 0) {
        OPENVINO_THROW("TypedArray.byteOffset has to be equal to zero.");
    }
    auto array_buffer = typed_array.ArrayBuffer();
    auto tensor = ov::Tensor(type, shape, array_buffer.Data());
    if (tensor.get_byte_size() != array_buffer.ByteLength()) {
        OPENVINO_THROW("Memory allocated using shape and element::type mismatch passed data's size");
    }
    return tensor;
}

void fill_tensor_from_strings(ov::Tensor& tensor, const Napi::Array& arr) {
    if (tensor.get_size() != static_cast<size_t>(arr.Length())) {
        OPENVINO_THROW("Passed array must have the same size (number of elements) as the Tensor!");
    }
    const auto data = tensor.data<std::string>();
    for (uint32_t i = 0; i < tensor.get_size(); ++i) {
        OPENVINO_ASSERT(arr[i].IsString(), "The array passed to create string tensor must contain only strings.");
        data[i] = arr[i].ToString().Utf8Value();
    }
}

ov::Tensor cast_to_tensor(const Napi::Array& array) {
    auto tensor = ov::Tensor(ov::element::string, ov::Shape{array.Length()});
    fill_tensor_from_strings(tensor, array);
    return tensor;
}

/**
 * @brief  Template function to convert C++ map into Javascript Object. Map key must be std::string.
 * @tparam MapElementType C++ data type of map elements.
 * @param info Contains the environment in which to construct a JavaScript object.
 * @return Napi::Object.
 */
template <typename MapElementType>
Napi::Object cpp_map_to_js_object(const Napi::CallbackInfo& info, const std::map<std::string, MapElementType>& map) {
    Napi::Object obj = Napi::Object::New(info.Env());

    for (const auto& [k, v] : map) {
        obj.Set(k, v);
    }

    return obj;
}

/**
 * @brief  Template function to convert C++ vector type into Javascript Array
 * @tparam SourceType C++ data type of vector elements.
 * @param info Contains the environment in which to construct a JavaScript object.
 * @return Napi::Array.
 */
template <typename SourceType>
Napi::Array cpp_vector_to_js_array(const Napi::CallbackInfo& info, const std::vector<SourceType>& vec) {
    auto array = Napi::Array::New(info.Env(), vec.size());

    uint32_t i = 0;
    for (auto& property : vec) {
        auto any = ov::Any(property);
        array[i++] = any_to_js(info, any);
    }

    return array;
}

Napi::Value any_to_js(const Napi::CallbackInfo& info, ov::Any value) {
    // Check for std::string
    if (value.is<std::string>()) {
        return Napi::String::New(info.Env(), value.as<std::string>());
    }
    // Check for bool
    else if (value.is<bool>()) {
        return Napi::Boolean::New(info.Env(), value.as<bool>());
    }
    // Check for ov::PropertyName
    else if (value.is<ov::PropertyName>()) {
        return Napi::String::New(info.Env(), value.as<std::string>());
    }
    // Check for ov::device::Type
    else if (value.is<ov::device::Type>()) {
        return Napi::String::New(info.Env(), value.as<std::string>());
    }
    // Check for int
    else if (value.is<int>()) {
        return Napi::Number::New(info.Env(), value.as<int>());
    }
    // Check for ov::Affinity
    else if (value.is<ov::Affinity>()) {
        return Napi::String::New(info.Env(), value.as<std::string>());
    }
    // Check for ov::element::Type
    else if (value.is<ov::element::Type>()) {
        return Napi::String::New(info.Env(), value.as<std::string>());
    }
    // Check for ov::hint::PerformanceMode
    else if (value.is<ov::hint::PerformanceMode>()) {
        return Napi::String::New(info.Env(), value.as<std::string>());
    }
    // Check for ov::hint::ExecutionMode
    else if (value.is<ov::hint::ExecutionMode>()) {
        return Napi::String::New(info.Env(), value.as<std::string>());
    }
    // Check for ov::hint::SchedulingCoreType
    else if (value.is<ov::hint::SchedulingCoreType>()) {
        return Napi::String::New(info.Env(), value.as<std::string>());
    }
    // Check for ov::log::Level
    else if (value.is<ov::log::Level>()) {
        return Napi::String::New(info.Env(), value.as<std::string>());
    }
    // Check for uint32_t
    else if (value.is<uint32_t>()) {
        return Napi::Number::New(info.Env(), value.as<uint32_t>());
    }
    // Check for std::vector<ov::Any>
    else if (value.is<const std::vector<ov::Any>>()) {
        auto p = value.as<const std::vector<ov::Any>>();

        return cpp_vector_to_js_array(info, p);
    }
    // Check for std::vector<ov::PropertyName>
    else if (value.is<const std::vector<ov::PropertyName>>()) {
        auto p = value.as<const std::vector<ov::PropertyName>>();

        return cpp_vector_to_js_array(info, p);
    }
    // Check for std::vector<std::string>
    else if (value.is<const std::vector<std::string>>()) {
        auto p = value.as<const std::vector<std::string>>();

        return cpp_vector_to_js_array(info, p);
    }
    // Check for std::vector<int>
    else if (value.is<std::vector<int>>()) {
        auto p = value.as<std::vector<int>>();

        return cpp_vector_to_js_array(info, p);
    }
    // Check for std::vector<int64_t>
    else if (value.is<std::vector<int64_t>>()) {
        auto p = value.as<std::vector<int64_t>>();

        return cpp_vector_to_js_array(info, p);
    }
    // Check for std::vector<unsigned int>
    else if (value.is<std::vector<unsigned int>>()) {
        auto p = value.as<std::vector<unsigned int>>();

        return cpp_vector_to_js_array(info, p);
    }
    // Check for std::vector<float>
    else if (value.is<std::vector<float>>()) {
        auto p = value.as<std::vector<float>>();

        return cpp_vector_to_js_array(info, p);
    }
    // Check for std::vector<double>
    else if (value.is<std::vector<double>>()) {
        auto p = value.as<std::vector<double>>();

        return cpp_vector_to_js_array(info, p);
    }
    // Check for std::tuple<unsigned int, unsigned int>
    else if (value.is<std::tuple<unsigned int, unsigned int>>()) {
        auto p = value.as<std::tuple<unsigned int, unsigned int>>();
        auto [first, second] = p;

        Napi::Array array = Napi::Array::New(info.Env(), 2);
        uint32_t indexes[] = {0, 1};

        array[indexes[0]] = Napi::Number::New(info.Env(), first);
        array[indexes[1]] = Napi::Number::New(info.Env(), second);

        return array;
    }
    // Check for std::tuple<unsigned int, unsigned int, unsigned int>
    else if (value.is<std::tuple<unsigned int, unsigned int, unsigned int>>()) {
        auto p = value.as<std::tuple<unsigned int, unsigned int, unsigned int>>();
        auto [first, second, third] = p;

        Napi::Array array = Napi::Array::New(info.Env(), 2);
        uint32_t indexes[] = {0, 1, 2};

        array[indexes[0]] = Napi::Number::New(info.Env(), first);
        array[indexes[1]] = Napi::Number::New(info.Env(), second);
        array[indexes[2]] = Napi::Number::New(info.Env(), third);

        return array;
    }
    // Check for std::map<std::string, std::string>
    else if (value.is<std::map<std::string, std::string>>()) {
        auto p = value.as<std::map<std::string, std::string>>();

        return cpp_map_to_js_object(info, p);
    }
    // Check for std::map<std::string, int>
    else if (value.is<std::map<std::string, int>>()) {
        auto p = value.as<std::map<std::string, int>>();

        return cpp_map_to_js_object(info, p);
    }
    // Check for std::map<std::string, uint64_t>
    else if (value.is<std::map<std::string, uint64_t>>()) {
        auto p = value.as<std::map<std::string, uint64_t>>();

        return cpp_map_to_js_object(info, p);
    }

    return info.Env().Undefined();
}

bool is_napi_value_int(const Napi::Env& env, const Napi::Value& num) {
    return env.Global().Get("Number").ToObject().Get("isInteger").As<Napi::Function>().Call({num}).ToBoolean().Value();
}

ov::AnyMap to_anyMap(const Napi::Env& env, const Napi::Value& val) {
    ov::AnyMap properties;
    if (!val.IsObject()) {
        OPENVINO_THROW("Passed Napi::Value must be an object.");
    }
    const auto& parameters = val.ToObject();
    const auto& keys = parameters.GetPropertyNames();

    for (uint32_t i = 0; i < keys.Length(); ++i) {
        const auto& property_name = static_cast<Napi::Value>(keys[i]).ToString().Utf8Value();

        const auto& any_value = js_to_cpp<ov::Any>(env, parameters.Get(property_name));

        properties.insert(std::make_pair(property_name, any_value));
    }

    return properties;
}

std::string buffer_to_string(const Napi::Value& value) {
    Napi::Buffer<uint8_t> model_data = value.As<Napi::Buffer<uint8_t>>();

    return std::string(reinterpret_cast<char*>(model_data.Data()), model_data.Length());
}
