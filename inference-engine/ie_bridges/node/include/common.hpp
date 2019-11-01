#include <napi.h>
#include <ie_core.hpp>

template<class T, class K>
const Napi::Array vector_to_napi_array(const Napi::Env& env, const std::vector<T> & vec);

template<class T, class K>
const Napi::Array tuple_to_napi_array(const Napi::Env& env, const std::tuple<T, T, T> & tpl);

template<class T, class K>
const Napi::Array tuple_to_napi_array(const Napi::Env& env, const std::tuple<T, T> & tpl);

Napi::Value parseParameter(const Napi::Env& env, const InferenceEngine::Parameter &param);
