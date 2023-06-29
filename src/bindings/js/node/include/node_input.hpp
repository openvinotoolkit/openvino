#pragma once

#include <napi.h>

#include <openvino/core/node_input.hpp>

class Input : public Napi::ObjectWrap<Input> {
public:
    Input(const Napi::CallbackInfo& info);

    /**
     * @brief Defines a Javascript Input class with constructor, static and instance properties and methods.
     * @param env The environment in which to construct a JavaScript class.
     * @return Napi::Function representing the constructor function for the Javascript Input class.
     */
    static Napi::Function GetClassConstructor(Napi::Env env);

    /// @brief This method is called during initialization of OpenVino native add-on.
    /// It exports JavaScript Input class.
    static Napi::Object Init(Napi::Env env, Napi::Object exports);

    void set_input(const std::shared_ptr<ov::Input<const ov::Node>>& input);

    static Napi::Object Wrap(Napi::Env env, std::shared_ptr<ov::Input<const ov::Node>> input);

private:
    std::shared_ptr<ov::Input<const ov::Node>> _input;
};
