#include "common.hpp"

Napi::Value parseParameter(const Napi::Env& env, const InferenceEngine::Parameter &param) {
    if (param.is<std::string>()) {
        return Napi::String::New(env, param.as<std::string>().c_str());
    }
    else if (param.is<int>()) {
        auto val = param.as<int>();
        return Napi::Number::New(env, static_cast<long>(val));
    }
    else if (param.is<unsigned int>()) {
        auto val = param.as<unsigned int>();
        return Napi::Number::New(env, static_cast<unsigned int>(val));
    }
    else if (param.is<float>()) {
        auto val = param.as<float>();
        return Napi::Number::New(env, static_cast<float>(val));
    }
    else if (param.is<bool>()) {
        auto val = param.as<bool>();
        return Napi::Boolean::New(env, static_cast<bool>(val));
    }

    else if (param.is<std::vector<std::string>>()) {
        auto vec = param.as<std::vector<std::string>>();
        return vector_to_napi_array<std::string, Napi::String >(env, vec);
    }

    else if (param.is<std::vector<int>>()){
        auto vec = param.as<std::vector<int>>();
        return vector_to_napi_array<int, Napi::Number>(env, vec);
    }

    else if (param.is<std::vector<unsigned int>>()){
        auto vec = param.as<std::vector<unsigned int>>();
        return vector_to_napi_array<unsigned int, Napi::Number>(env, vec);
    }

    else if (param.is<std::tuple<int, int >>()) {
        auto tpl = param.as<std::tuple<int, int >>();
        return tuple_to_napi_array<int, Napi::Number>(env, tpl);
    }

    else if (param.is<std::tuple<int, int, int >>()) {
        auto tpl = param.as<std::tuple<int, int, int >>();
        return tuple_to_napi_array<int, Napi::Number>(env, tpl);
    }

    else if (param.is<std::tuple<unsigned int, unsigned int >>()) {
        auto tpl = param.as<std::tuple<unsigned int, unsigned int >>();
        return tuple_to_napi_array<unsigned int, Napi::Number>(env, tpl);
    }

    else if (param.is<std::tuple<unsigned int, unsigned int, unsigned int >>()) {
        auto tpl = param.as<std::tuple<unsigned int, unsigned int, unsigned int >>();
        return tuple_to_napi_array<unsigned int, Napi::Number>(env, tpl);
    }
    else if (param.is<std::map<std::string, std::string>>()) {
        auto tpl = param.as<std::tuple<std::string, std::string >>();
        return tuple_to_napi_array<std::string, Napi::String>(env, tpl);
    }

    else if (param.is<std::map<std::string, int>>()) {
        auto dict = param.as<std::map<std::string, int>>();
        Napi::Array result = Napi::Array::New(env);
        auto i = 0;
        for (const auto &it : dict){
            auto obj = Napi::Object::New(env);
            obj.Set("name", it.first.c_str());
            obj.Set("value",Napi::Number::New(env, static_cast<int>(it.second)));
            result[i] = obj;
            i++;
        }
        return result;
    }

    throw Napi::Error::New(env, "Cannot parse argument");
}

template<class T, class K>
const Napi::Array vector_to_napi_array(const Napi::Env& env, const std::vector<T> & vec) {
    Napi::Array result = Napi::Array::New(env);
    int size = vec.size();
    for (int i = 0; i < size; i++) {
        result[i] = K::New(env, vec[i]);
    }
    return result;
}

template<class T, class K>
const Napi::Array tuple_to_napi_array(const Napi::Env& env, const std::tuple<T, T, T> & tpl) {
    Napi::Array result = Napi::Array::New(env, 3);
    int i = 0;
    result[i++] = K::New(env, static_cast<T>(std::get<0>(tpl)));
    result[i++] = K::New(env, static_cast<T>(std::get<1>(tpl)));
    result[i++] = K::New(env, static_cast<T>(std::get<2>(tpl)));
    return result;
}

template<class T, class K>
const Napi::Array tuple_to_napi_array(const Napi::Env& env, const std::tuple<T, T> & tpl) {
    Napi::Array result = Napi::Array::New(env, 2);
    int i = 0;
    result[i++] = K::New(env, static_cast<T>(std::get<0>(tpl)));
    result[i++] = K::New(env, static_cast<T>(std::get<1>(tpl)));
    return result;
}