#include "pre_post_process_wrap.hpp"

#include <iostream>

PrePostProcessorWrap::PrePostProcessorWrap(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<PrePostProcessorWrap>(info) {
    if (info.Length() != 1)
        reportError(info.Env(), "Invalid number of arguments for PrePostProcessor constructor.");
    else {
        Napi::Object obj = info[0].ToObject();
        auto* m = Napi::ObjectWrap<ModelWrap>::Unwrap(obj);
        _ppp = std::unique_ptr<ov::preprocess::PrePostProcessor>(new ov::preprocess::PrePostProcessor(m->get_model()));
    }
}

Napi::Function PrePostProcessorWrap::GetClassConstructor(Napi::Env env) {
    return DefineClass(env,
                       "PrePostProcessorWrap",
                       {InstanceMethod("set_input_tensor_shape", &PrePostProcessorWrap::set_input_tensor_shape),
                        InstanceMethod("set_input_tensor_layout", &PrePostProcessorWrap::set_input_tensor_layout),
                        InstanceMethod("set_input_model_layout", &PrePostProcessorWrap::set_input_model_layout),
                        InstanceMethod("build", &PrePostProcessorWrap::build)});
}

Napi::Object PrePostProcessorWrap::Init(Napi::Env env, Napi::Object exports) {
    auto func = GetClassConstructor(env);

    Napi::FunctionReference* constructor = new Napi::FunctionReference();
    *constructor = Napi::Persistent(func);
    env.SetInstanceData(constructor);

    exports.Set("PrePostProcessor", func);
    return exports;
}

Napi::Value PrePostProcessorWrap::set_input_tensor_shape(const Napi::CallbackInfo& info) {
    auto shape = js_to_cpp<ov::Shape>(info, 0, {napi_int32_array, js_array});
    _ppp->input().tensor().set_shape(shape);
    return info.This();
}

Napi::Value PrePostProcessorWrap::set_input_tensor_layout(const Napi::CallbackInfo& info) {
    auto layout = js_to_cpp<ov::Layout>(info, 0, {napi_string});
    _ppp->input().tensor().set_layout(layout);
    return info.This();
}

Napi::Value PrePostProcessorWrap::set_input_model_layout(const Napi::CallbackInfo& info) {
    auto layout = js_to_cpp<ov::Layout>(info, 0, {napi_string});
    _ppp->input().model().set_layout(layout);
    return info.This();
}

Napi::Value PrePostProcessorWrap::build(const Napi::CallbackInfo& info) {
    _ppp->build();
    return info.This();
}
