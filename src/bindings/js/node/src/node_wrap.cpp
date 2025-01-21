#include "node_wrap.h"

Napi::FunctionReference NodeWrap::constructor;

//NodeWrap
NodeWrap::NodeWrap(const Napi::CallbackInfo& info, ov::Node* node) : Napi::ObjectWrap<NodeWrap>(info) {
    this->node_ = node;
}

Napi::Value NodeWrap::get_name(const Napi::CallbackInfo& info) {
    return Napi::String::New(info.Env(), this->node_->get_name());
}

Napi::Object NodeWrap::New(Napi::Env env, ov::Node* node) {
    Napi::Object obj = Napi::ObjectWrap<NodeWrap>::New(env, node);
    return obj;
}

Napi::Object Init(Napi::Env env, Napi::Object exports) {
    Napi::Function func = DefineClass(env, "Node", {
      InstanceMethod("getName", &NodeWrap::get_name),
    }, [](const Napi::CallbackInfo& info) {
        ov::Node* node = info[0].As<Napi::External<ov::Node>>().Data();
        return NodeWrap::New(env, node);
    });

    constructor = Napi::Persistent(func);
    exports.Set("Node", func);
    return exports;
}

NODE_API_MODULE(Node, Init)