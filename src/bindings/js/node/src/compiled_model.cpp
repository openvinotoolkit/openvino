#include "compiled_model.hpp"

#include "infer_request.hpp"
#include "node_output.hpp"

CompiledModelWrap::CompiledModelWrap(const Napi::CallbackInfo& info) : Napi::ObjectWrap<CompiledModelWrap>(info) {}

Napi::Function CompiledModelWrap::GetClassConstructor(Napi::Env env) {
    return DefineClass(env,
                       "CompiledModel",
                       {InstanceMethod("create_infer_request", &CompiledModelWrap::create_infer_request),
                        InstanceAccessor<&CompiledModelWrap::get_inputs>("inputs"),
                        InstanceAccessor<&CompiledModelWrap::get_outputs>("outputs")});
}

Napi::Object CompiledModelWrap::Init(Napi::Env env, Napi::Object exports) {
    auto func = GetClassConstructor(env);

    Napi::FunctionReference* constructor = new Napi::FunctionReference();
    *constructor = Napi::Persistent(func);
    env.SetInstanceData(constructor);

    exports.Set("CompiledModel", func);
    return exports;
}

Napi::Object CompiledModelWrap::Wrap(Napi::Env env, ov::CompiledModel compiled_model) {
    Napi::HandleScope scope(env);
    Napi::Object obj = GetClassConstructor(env).New({});
    CompiledModelWrap* cm = Napi::ObjectWrap<CompiledModelWrap>::Unwrap(obj);
    cm->_compiled_model = compiled_model;
    return obj;
}

Napi::Value CompiledModelWrap::create_infer_request(const Napi::CallbackInfo& info) {
    ov::InferRequest infer_request = _compiled_model.create_infer_request();
    return InferRequestWrap::Wrap(info.Env(), infer_request);
}

Napi::Value CompiledModelWrap::get_outputs(const Napi::CallbackInfo& info) {
    auto cm_outputs = _compiled_model.outputs();  // Output<Node>
    Napi::Array js_outputs = Napi::Array::New(info.Env(), cm_outputs.size());

    size_t i = 0;
    for (auto& out : cm_outputs)
        js_outputs[i++] = Output::Wrap(info.Env(), out);

    return js_outputs;
}

Napi::Value CompiledModelWrap::get_inputs(const Napi::CallbackInfo& info) {
    auto cm_inputs = _compiled_model.inputs();  // Output<Node>
    Napi::Array js_inputs = Napi::Array::New(info.Env(), cm_inputs.size());

    size_t i = 0;
    for (auto& out : cm_inputs)
        js_inputs[i++] = Output::Wrap(info.Env(), out);

    return js_inputs;
}
