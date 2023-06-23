#pragma once

#include <napi.h>

#include <openvino/core/node_output.hpp>

class Output : public Napi::ObjectWrap<Output> {
public:
    /**
     * @brief TO_DO
     */
    Output(const Napi::CallbackInfo& info);

    /**
     * @brief Defines a Javascript Output class with constructor, static and instance properties and methods.
     * @param env The environment in which to construct a JavaScript class.
     * @return Napi::Function representing the constructor function for the Javascript CompiledModel class.
     */
    static Napi::Function GetClassConstructor(Napi::Env env);

    /// @brief This method is called during initialization of OpenVino native add-on.
    /// It exports JavaScript Output class.
    static Napi::Object Init(Napi::Env env, Napi::Object exports);

    void set_output(const ov::Output<const ov::Node>& output);

    static Napi::Object Wrap(Napi::Env env, ov::Output<const ov::Node> output);

    Napi::Value get_any_name(const Napi::CallbackInfo& info);

private:
    ov::Output<const ov::Node> _output;
};
