#include "InferRequestWrap.hpp"

#include "TensorWrap.hpp"

InferRequestWrap::InferRequestWrap(const Napi::CallbackInfo& info) : Napi::ObjectWrap<InferRequestWrap>(info) {}

Napi::Function InferRequestWrap::GetClassConstructor(Napi::Env env) {
    return DefineClass(env,
                       "InferRequest",
                       {InstanceMethod("set_input_tensor", &InferRequestWrap::set_input_tensor),
                        InstanceMethod("infer", &InferRequestWrap::infer),
                        InstanceMethod("get_output_tensor", &InferRequestWrap::get_output_tensor)});
}

Napi::Object InferRequestWrap::Init(Napi::Env env, Napi::Object exports) {
    auto func = GetClassConstructor(env);

    Napi::FunctionReference* constructor = new Napi::FunctionReference();
    *constructor = Napi::Persistent(func);
    env.SetInstanceData(constructor);

    exports.Set("InferRequest", func);
    return exports;
}

void InferRequestWrap::set_infer_request(ov::InferRequest& infer_request) {
    _infer_request = infer_request;
}

Napi::Object InferRequestWrap::Wrap(Napi::Env env, ov::InferRequest infer_request) {
    Napi::HandleScope scope(env);
    Napi::Object obj = GetClassConstructor(env).New({});
    InferRequestWrap* ir = Napi::ObjectWrap<InferRequestWrap>::Unwrap(obj);
    ir->set_infer_request(infer_request);
    return obj;
}

Napi::Value InferRequestWrap::set_input_tensor(const Napi::CallbackInfo& info) {
    auto* tensorWrap = Napi::ObjectWrap<TensorWrap>::Unwrap(info[0].ToObject());
    ov::Tensor t = tensorWrap->get_tensor();

    // ov::Tensor t = TensorWrap::get_tensor(info);
    _infer_request.set_input_tensor(t);
    return Napi::Value();
}

Napi::Value InferRequestWrap::infer(const Napi::CallbackInfo& info) {
    _infer_request.infer();
    return Napi::Value();
}

Napi::Value InferRequestWrap::get_output_tensor(const Napi::CallbackInfo& info) {
    ov::Tensor tensor = _infer_request.get_output_tensor();
    return TensorWrap::Wrap(info.Env(), tensor);
}