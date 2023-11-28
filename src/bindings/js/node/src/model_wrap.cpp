// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "model_wrap.hpp"

#include "addon.hpp"
#include "node_output.hpp"

ModelWrap::ModelWrap(const Napi::CallbackInfo& info) : Napi::ObjectWrap<ModelWrap>(info) {}

Napi::Function ModelWrap::GetClassConstructor(Napi::Env env) {
    return DefineClass(env,
                       "ModelWrap",
                       {InstanceMethod("getName", &ModelWrap::get_name),
                        InstanceMethod("output", &ModelWrap::get_output),
                        InstanceMethod("input", &ModelWrap::get_input),
                        InstanceAccessor<&ModelWrap::get_inputs>("inputs"),
                        InstanceAccessor<&ModelWrap::get_outputs>("outputs")});
}

Napi::Object ModelWrap::Init(Napi::Env env, Napi::Object exports) {
    Napi::Function prototype = GetClassConstructor(env);

    Napi::FunctionReference* constructor = new Napi::FunctionReference();
    *constructor = Napi::Persistent(prototype);
    auto data = env.GetInstanceData<AddonData>();
    data->model_prototype = constructor;

    exports.Set("Model", prototype);
    return exports;
}

void ModelWrap::set_model(const std::shared_ptr<ov::Model>& model) {
    _model = model;
}

Napi::Object ModelWrap::Wrap(Napi::Env env, std::shared_ptr<ov::Model> model) {
    Napi::HandleScope scope(env);
    auto obj = env.GetInstanceData<AddonData>()->model_prototype->New({});
    ModelWrap* m = Napi::ObjectWrap<ModelWrap>::Unwrap(obj);
    m->set_model(model);
    return obj;
}

Napi::Value ModelWrap::get_name(const Napi::CallbackInfo& info) {
    if (_model->get_name() != "")
        return Napi::String::New(info.Env(), _model->get_name());
    else
        return Napi::String::New(info.Env(), "unknown");
}

std::string ModelWrap::get_name() {
    if (_model->get_name() != "")
        return _model->get_name();
    else
        return "unknown";
}

std::shared_ptr<ov::Model> ModelWrap::get_model() {
    return _model;
}

Napi::Value ModelWrap::get_input(const Napi::CallbackInfo& info) {
    if (info.Length() == 0) {
        try {
            return Output<ov::Node>::Wrap(info.Env(), _model->input());
        } catch (std::exception& e) {
            reportError(info.Env(), e.what());
            return Napi::Value();
        }
    } else if (info.Length() != 1) {
        reportError(info.Env(), "Invalid number of arguments -> " + std::to_string(info.Length()));
        return Napi::Value();
    } else if (info[0].IsString()) {
        const auto& tensor_name = info[0].ToString();
        return Output<ov::Node>::Wrap(info.Env(), _model->input(tensor_name));
    } else if (info[0].IsNumber()) {
        const auto& idx = info[0].As<Napi::Number>().Int32Value();
        return Output<ov::Node>::Wrap(info.Env(), _model->input(idx));
    } else {
        reportError(info.Env(), "Error while getting model outputs.");
        return info.Env().Undefined();
    }
}

Napi::Value ModelWrap::get_output(const Napi::CallbackInfo& info) {
    if (info.Length() == 0) {
        try {
            return Output<ov::Node>::Wrap(info.Env(), _model->output());
        } catch (std::exception& e) {
            reportError(info.Env(), e.what());
            return Napi::Value();
        }
    } else if (info.Length() != 1) {
        reportError(info.Env(), "Invalid number of arguments -> " + std::to_string(info.Length()));
        return Napi::Value();
    } else if (info[0].IsString()) {
        auto tensor_name = info[0].ToString();
        return Output<ov::Node>::Wrap(info.Env(), _model->output(tensor_name));
    } else if (info[0].IsNumber()) {
        auto idx = info[0].As<Napi::Number>().Int32Value();
        return Output<ov::Node>::Wrap(info.Env(), _model->output(idx));
    } else {
        reportError(info.Env(), "Error while getting model outputs.");
        return Napi::Value();
    }
}

Napi::Value ModelWrap::get_inputs(const Napi::CallbackInfo& info) {
    auto cm_inputs = _model->inputs();  // Output<Node>
    Napi::Array js_inputs = Napi::Array::New(info.Env(), cm_inputs.size());

    size_t i = 0;
    for (auto& input : cm_inputs)
        js_inputs[i++] = Output<ov::Node>::Wrap(info.Env(), input);

    return js_inputs;
}

Napi::Value ModelWrap::get_outputs(const Napi::CallbackInfo& info) {
    auto cm_outputs = _model->outputs();  // Output<Node>
    Napi::Array js_outputs = Napi::Array::New(info.Env(), cm_outputs.size());

    size_t i = 0;
    for (auto& out : cm_outputs)
        js_outputs[i++] = Output<ov::Node>::Wrap(info.Env(), out);

    return js_outputs;
}
