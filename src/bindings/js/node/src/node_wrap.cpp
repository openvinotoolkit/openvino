#include "node_wrap.h"

// Initialize the static member
Napi::FunctionReference NodeWrap::constructor;

// NodeWrap constructor
NodeWrap::NodeWrap(const Napi::CallbackInfo &info, ov::Node *node)
    : Napi::ObjectWrap<NodeWrap>(info), node_(node) {
  assert(node_ != nullptr && "Node pointer cannot be null");
}

Napi::Value NodeWrap::get_name(const Napi::CallbackInfo &info) const {
  assert(node_ != nullptr && "Node pointer cannot be null");
  return Napi::String::New(info.Env(), this->node_->get_name());
}

// Return the Napi::Function that represents the Node class
Napi::Function NodeWrap::get_class(Napi::Env env) {
  if (constructor.IsEmpty()) {
    throw Napi::Error::New(
        env, "NodeWrap class is not initialized. Call Init first.");
  }
  return constructor.Value();
}

Napi::Object NodeWrap::New(Napi::Env env, ov::Node *node) {
  // Ensure node is not null before creating the object
  assert(node != nullptr && "Node pointer cannot be null");
  Napi::Object obj = constructor.New({});
  NodeWrap *wrap = Napi::ObjectWrap<NodeWrap>::Unwrap(obj);
  wrap->node_ = node;
  return obj;
}

Napi::Object Init(Napi::Env env, Napi::Object exports) {
  Napi::Function func =
      DefineClass(env, "Node",
                  {
                      InstanceMethod("getName", &NodeWrap::get_name),
                  });

  // Store the class constructor in the static member
  NodeWrap::constructor = Napi::Persistent(func);
  NodeWrap::constructor.SuppressDestruct();

  exports.Set("Node", func);
  return exports;
}

NODE_API_MODULE(Node, Init)