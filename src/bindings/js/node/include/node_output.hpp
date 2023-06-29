#pragma once

#include <napi.h>

#include <openvino/core/node_output.hpp>

template <class NodeType>
class Output : public Napi::ObjectWrap<Output<NodeType>> {
public:
    /**
     * @brief TO_DO
     */
    Output(const Napi::CallbackInfo& info) : Napi::ObjectWrap<Output<NodeType>>(info) {}

    /**
     * @brief Defines a Javascript Output class with constructor, static and instance properties and methods.
     * @param env The environment in which to construct a JavaScript class.
     * @return Napi::Function representing the constructor function for the Javascript Output class.
     */
    static Napi::Function GetClassConstructor(Napi::Env env) {
        return Output::DefineClass(env,
                                   "Output",
                                   {Output::InstanceMethod("getAnyName", &Output::get_any_name),
                                    Output::InstanceMethod("toString", &Output::get_any_name)});
    }

    /// @brief This method is called during initialization of OpenVino native add-on.
    /// It exports JavaScript Output class.
    static Napi::Object Init(Napi::Env env, Napi::Object exports) {
        auto func = GetClassConstructor(env);

        Napi::FunctionReference* constructor = new Napi::FunctionReference();
        *constructor = Napi::Persistent(func);
        env.SetInstanceData(constructor);

        exports.Set("Output", func);
        return exports;
    }

    ov::Output<NodeType> get_output() const {
        return _output;
    }

    static Napi::Object Wrap(Napi::Env env, ov::Output<NodeType> output) {
        Napi::HandleScope scope(env);
        Napi::Object obj = GetClassConstructor(env).New({});
        Output* output_ptr = Napi::ObjectWrap<Output<NodeType>>::Unwrap(obj);
        output_ptr->_output = output;
        return obj;
    }

    Napi::Value get_any_name(const Napi::CallbackInfo& info) {
        return Napi::String::New(info.Env(), _output.get_any_name());
    }

private:
    // ov::Output<const ov::Node> _output;
    ov::Output<NodeType> _output;
};
