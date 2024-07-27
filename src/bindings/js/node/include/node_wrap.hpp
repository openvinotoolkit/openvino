#pragma once
#include <napi.h>

#include <openvino/openvino.hpp>

class NodeWrap : public Napi::ObjectWrap<NodeWrap> {
public:
    /**
     * @brief Constructs NodeWrap from the Napi::CallbackInfo.
     * @param info contains passed arguments. Can be empty.
     */
    NodeWrap(const Napi::CallbackInfo& info);
    /**
     * @brief Defines a JavaScript NodeWrap class with constructor, static, and instance properties and methods.
     * @param env The environment in which to construct a JavaScript class.
     * @return Napi::Function representing the constructor function for the JavaScript NodeWrap class.
     */
    static Napi::Function get_class(Napi::Env env);
    /**
     * @brief Helper function to access the name of the node.
     * @param info Contains information about the environment and passed arguments.
     * @return Napi::String containing the node's name.
     */
    Napi::Value get_name(const Napi::CallbackInfo& info);

private:
    static Napi::FunctionReference constructor;
    std::shared_ptr<ov::Node> node;
};
