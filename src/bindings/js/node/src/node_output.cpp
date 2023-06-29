// #include "node_output.hpp"

// Output::Output(const Napi::CallbackInfo& info) : Napi::ObjectWrap<Output>(info) {}

// Napi::Function Output::GetClassConstructor(Napi::Env env) {
//     return DefineClass(
//         env,
//         "Output",
//         {InstanceMethod("getAnyName", &Output::get_any_name), InstanceMethod("toString", &Output::get_any_name)});
// }

// Napi::Object Output::Init(Napi::Env env, Napi::Object exports) {
//     auto func = GetClassConstructor(env);

//     Napi::FunctionReference* constructor = new Napi::FunctionReference();
//     *constructor = Napi::Persistent(func);
//     env.SetInstanceData(constructor);

//     exports.Set("Output", func);
//     return exports;
// }

// ov::Output<const ov::Node> Output::get_output() const {
//     return _output;
// }

// Napi::Object Output::Wrap(Napi::Env env, ov::Output<const ov::Node> output) {
//     Napi::HandleScope scope(env);
//     Napi::Object obj = GetClassConstructor(env).New({});
//     Output* output_ptr = Napi::ObjectWrap<Output>::Unwrap(obj);
//     output_ptr->_output = output;
//     return obj;
// }

// Napi::Value Output::get_any_name(const Napi::CallbackInfo& info) {
//     return Napi::String::New(info.Env(), _output.get_any_name());
// }
