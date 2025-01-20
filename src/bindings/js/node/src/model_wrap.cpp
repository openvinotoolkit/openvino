// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "node/include/model_wrap.hpp"

#include "node/include/addon.hpp"
#include "node/include/errors.hpp"
#include "node/include/helper.hpp"
#include "node/include/node_output.hpp"
#include "node/include/type_validation.hpp"

ModelWrap::ModelWrap(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<ModelWrap>(info),
      _model{},
      _core{},
      _compiled_model{} {}

Napi::Function ModelWrap::get_class(Napi::Env env) {
    return DefineClass(env,
                       "ModelWrap",
                       {InstanceMethod("getName", &ModelWrap::get_name),
                        InstanceMethod("output", &ModelWrap::get_output),
                        InstanceMethod("input", &ModelWrap::get_input),
                        InstanceMethod("isDynamic", &ModelWrap::is_dynamic),
                        InstanceMethod("getOutputSize", &ModelWrap::get_output_size),
                        InstanceMethod("setFriendlyName", &ModelWrap::set_friendly_name),
                        InstanceMethod("getFriendlyName", &ModelWrap::get_friendly_name),
                        InstanceMethod("getOutputShape", &ModelWrap::get_output_shape),
                        InstanceMethod("getOutputElementType", &ModelWrap::get_output_element_type),
                        InstanceMethod("clone", &ModelWrap::clone),
                        InstanceAccessor<&ModelWrap::get_inputs>("inputs"),
                        InstanceAccessor<&ModelWrap::get_outputs>("outputs")});
}

void ModelWrap::set_model(const std::shared_ptr<ov::Model>& model) {
    _model = model;
}

Napi::Value ModelWrap::get_name(const Napi::CallbackInfo& info) {
    if (_model->get_name() != "")
        return Napi::String::New(info.Env(), _model->get_name());
    else
        return Napi::String::New(info.Env(), "unknown");
}

std::shared_ptr<ov::Model> ModelWrap::get_model() const {
    return _model;
}

Napi::Value ModelWrap::get_input(const Napi::CallbackInfo& info) {
    if (info.Length() == 0) {
        try {
            return Output<ov::Node>::wrap(info.Env(), _model->input());
        } catch (std::exception& e) {
            reportError(info.Env(), e.what());
            return Napi::Value();
        }
    } else if (info.Length() != 1) {
        reportError(info.Env(), "Invalid number of arguments -> " + std::to_string(info.Length()));
        return Napi::Value();
    } else if (info[0].IsString()) {
        const auto& tensor_name = info[0].ToString();
        return Output<ov::Node>::wrap(info.Env(), _model->input(tensor_name));
    } else if (info[0].IsNumber()) {
        const auto& idx = info[0].As<Napi::Number>().Int32Value();
        return Output<ov::Node>::wrap(info.Env(), _model->input(idx));
    } else {
        reportError(info.Env(), "Error while getting model outputs.");
        return info.Env().Undefined();
    }
}

Napi::Value ModelWrap::get_output(const Napi::CallbackInfo& info) {
    if (info.Length() == 0) {
        try {
            return Output<ov::Node>::wrap(info.Env(), _model->output());
        } catch (std::exception& e) {
            reportError(info.Env(), e.what());
            return Napi::Value();
        }
    } else if (info.Length() != 1) {
        reportError(info.Env(), "Invalid number of arguments -> " + std::to_string(info.Length()));
        return Napi::Value();
    } else if (info[0].IsString()) {
        auto tensor_name = info[0].ToString();
        return Output<ov::Node>::wrap(info.Env(), _model->output(tensor_name));
    } else if (info[0].IsNumber()) {
        auto idx = info[0].As<Napi::Number>().Int32Value();
        return Output<ov::Node>::wrap(info.Env(), _model->output(idx));
    } else {
        reportError(info.Env(), "Error while getting model outputs.");
        return Napi::Value();
    }
}

Napi::Value ModelWrap::get_inputs(const Napi::CallbackInfo& info) {
    auto cm_inputs = _model->inputs();  // Output<Node>
    Napi::Array js_inputs = Napi::Array::New(info.Env(), cm_inputs.size());

    uint32_t i = 0;
    for (auto& input : cm_inputs)
        js_inputs[i++] = Output<ov::Node>::wrap(info.Env(), input);

    return js_inputs;
}

Napi::Value ModelWrap::get_outputs(const Napi::CallbackInfo& info) {
    auto cm_outputs = _model->outputs();  // Output<Node>
    Napi::Array js_outputs = Napi::Array::New(info.Env(), cm_outputs.size());

    uint32_t i = 0;
    for (auto& out : cm_outputs)
        js_outputs[i++] = Output<ov::Node>::wrap(info.Env(), out);

    return js_outputs;
}

Napi::Value ModelWrap::is_dynamic(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() > 0) {
        reportError(env, "isDynamic() does not accept any arguments.");
        return env.Null();
    }
    const auto result = _model->is_dynamic();
    return Napi::Boolean::New(env, result);
}

Napi::Value ModelWrap::get_output_size(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() > 0) {
        reportError(env, "getOutputSize() does not accept any arguments.");
        return env.Undefined();
    }
    const auto size = static_cast<double>(_model->get_output_size());
    return Napi::Number::New(env, size);
}

Napi::Value ModelWrap::set_friendly_name(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    try {
        if (info.Length() != 1 || !info[0].IsString()) {
            OPENVINO_THROW("Expected a single string argument for the friendly name");
        }
        const auto name = info[0].As<Napi::String>().Utf8Value();
        _model->set_friendly_name(name);
    } catch (const std::exception& e) {
        reportError(env, e.what());
    }
    return env.Undefined();
}

Napi::Value ModelWrap::get_friendly_name(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() > 0) {
        reportError(env, "getFriendlyName() does not take any arguments");
        return env.Undefined();
    }
    const auto friendly_name = _model->get_friendly_name();
    return Napi::String::New(env, friendly_name);
}

Napi::Value ModelWrap::get_output_shape(const Napi::CallbackInfo& info) {
    if (info.Length() != 1 || !info[0].IsNumber()) {
        reportError(info.Env(), "Invalid argument. Expected a single number.");
        return info.Env().Undefined();
    }

    try {
        auto idx = info[0].As<Napi::Number>().Int32Value();
        auto output = _model->output(idx);
        return cpp_to_js<ov::Shape, Napi::Array>(info, output.get_shape());
    } catch (const std::exception& e) {
        reportError(info.Env(), e.what());
        return info.Env().Undefined();
    }
}

Napi::Value ModelWrap::get_output_element_type(const Napi::CallbackInfo& info) {
    std::vector<std::string> allowed_signatures;
    try {
        if (ov::js::validate<int>(info, allowed_signatures)) {
            auto idx = info[0].As<Napi::Number>().Int32Value();
            const auto& output = _model->output(idx);
            return cpp_to_js<ov::element::Type_t, Napi::String>(info, output.get_element_type());
        } else {
            OPENVINO_THROW("'getOutputElementType'", ov::js::get_parameters_error_msg(info, allowed_signatures));
        }
    } catch (const std::exception& e) {
        reportError(info.Env(), e.what());
        return info.Env().Undefined();
    }
}

Napi::Value ModelWrap::clone(const Napi::CallbackInfo& info) {
    std::vector<std::string> allowed_signatures;
    try {
        if (ov::js::validate(info, allowed_signatures)) {
            return cpp_to_js(info.Env(), _model->clone());
        } else {
            OPENVINO_THROW("'clone'", ov::js::get_parameters_error_msg(info, allowed_signatures));
        }
    } catch (const std::exception& e) {
        reportError(info.Env(), e.what());
        return info.Env().Undefined();
    }
}
