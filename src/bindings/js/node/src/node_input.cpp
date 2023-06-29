#include "node_input.hpp"

Input::Input(const Napi::CallbackInfo& info) : Napi::ObjectWrap<Input>(info) {}

Napi::Function Input::GetClassConstructor(Napi::Env env) {
    return DefineClass(env,
                       "Input",
                       {

                       });
}

Napi::Object Input::Init(Napi::Env env, Napi::Object exports) {
    auto func = GetClassConstructor(env);

    Napi::FunctionReference* constructor = new Napi::FunctionReference();
    *constructor = Napi::Persistent(func);
    env.SetInstanceData(constructor);

    exports.Set("Input", func);
    return exports;
}

void Input::set_input(const std::shared_ptr<ov::Input<const ov::Node>>& input) {
    _input = input;
}

Napi::Object Input::Wrap(Napi::Env env, std::shared_ptr<ov::Input<const ov::Node>> input) {
    Napi::HandleScope scope(env);
    Napi::Object obj = GetClassConstructor(env).New({});
    Input* input_ptr = Napi::ObjectWrap<Input>::Unwrap(obj);
    input_ptr->set_input(input);
    return obj;
}
