#include "node_output.hpp"

Output::Output(const Napi::CallbackInfo& info) : Napi::ObjectWrap<Output>(info) {}

Napi::Function Output::GetClassConstructor(Napi::Env env) {
    return DefineClass(env, "Output", {});
}

Napi::Object Output::Init(Napi::Env env, Napi::Object exports) {
    auto func = GetClassConstructor(env);

    Napi::FunctionReference* constructor = new Napi::FunctionReference();
    *constructor = Napi::Persistent(func);
    env.SetInstanceData(constructor);

    exports.Set("Output", func);
    return exports;
}
