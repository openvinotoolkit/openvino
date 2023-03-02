#include "CompiledModelWrap.hpp"

#include "InferRequestWrap.hpp"

CompiledModelWrap::CompiledModelWrap(const Napi::CallbackInfo& info) : Napi::ObjectWrap<CompiledModelWrap>(info) {}

Napi::Function CompiledModelWrap::GetClassConstructor(Napi::Env env) {
    return DefineClass(env,
                       "CompiledModel",
                       {InstanceMethod("create_infer_request", &CompiledModelWrap::create_infer_request)});
}

Napi::Object CompiledModelWrap::Init(Napi::Env env, Napi::Object exports) {
    auto func = GetClassConstructor(env);

    Napi::FunctionReference* constructor = new Napi::FunctionReference();
    *constructor = Napi::Persistent(func);
    env.SetInstanceData(constructor);

    exports.Set("CompiledModel", func);
    return exports;
}

void CompiledModelWrap::set_compiled_model(const ov::CompiledModel& compiled_model) {
    _compiled_model = compiled_model;
}

Napi::Object CompiledModelWrap::Wrap(Napi::Env env, ov::CompiledModel compiled_model) {
    Napi::HandleScope scope(env);
    Napi::Object obj = GetClassConstructor(env).New({});
    CompiledModelWrap* cm = Napi::ObjectWrap<CompiledModelWrap>::Unwrap(obj);
    cm->set_compiled_model(compiled_model);
    return obj;
}

Napi::Value CompiledModelWrap::create_infer_request(const Napi::CallbackInfo& info) {
    ov::InferRequest infer_request = _compiled_model.create_infer_request();
    return InferRequestWrap::Wrap(info.Env(), infer_request);
}
