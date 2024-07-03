// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "node/include/compiled_model.hpp"

#include "node/include/addon.hpp"
#include "node/include/errors.hpp"
#include "node/include/infer_request.hpp"
#include "node/include/node_output.hpp"

CompiledModelWrap::CompiledModelWrap(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<CompiledModelWrap>(info),
      _compiled_model{} {}

Napi::Function CompiledModelWrap::get_class(Napi::Env env) {
    return DefineClass(env,
                       "CompiledModel",
                       {InstanceMethod("createInferRequest", &CompiledModelWrap::create_infer_request),
                        InstanceMethod("input", &CompiledModelWrap::get_input),
                        InstanceAccessor<&CompiledModelWrap::get_inputs>("inputs"),
                        InstanceMethod("output", &CompiledModelWrap::get_output),
                        InstanceAccessor<&CompiledModelWrap::get_outputs>("outputs"),
                        InstanceMethod("exportModelSync", &CompiledModelWrap::export_model)});
}

Napi::Object CompiledModelWrap::wrap(Napi::Env env, ov::CompiledModel compiled_model) {
    Napi::HandleScope scope(env);
    const auto& prototype = env.GetInstanceData<AddonData>()->compiled_model;
    if (!prototype) {
        OPENVINO_THROW("Invalid pointer to CompiledModel prototype.");
    }
    auto obj = prototype.New({});
    const auto cm = Napi::ObjectWrap<CompiledModelWrap>::Unwrap(obj);
    cm->_compiled_model = compiled_model;
    return obj;
}

void CompiledModelWrap::set_compiled_model(const ov::CompiledModel& compiled_model) {
    _compiled_model = compiled_model;
}

Napi::Value CompiledModelWrap::create_infer_request(const Napi::CallbackInfo& info) {
    ov::InferRequest infer_request = _compiled_model.create_infer_request();
    return InferRequestWrap::wrap(info.Env(), infer_request);
}

Napi::Value CompiledModelWrap::get_output(const Napi::CallbackInfo& info) {
    try {
        return get_node(
            info,
            static_cast<const ov::Output<const ov::Node>& (ov::CompiledModel::*)() const>(&ov::CompiledModel::output),
            static_cast<const ov::Output<const ov::Node>& (ov::CompiledModel::*)(const std::string&) const>(
                &ov::CompiledModel::output),
            static_cast<const ov::Output<const ov::Node>& (ov::CompiledModel::*)(size_t) const>(
                &ov::CompiledModel::output));
    } catch (std::exception& e) {
        reportError(info.Env(), e.what() + std::string("outputs."));
        return info.Env().Null();
    }
}

Napi::Value CompiledModelWrap::get_outputs(const Napi::CallbackInfo& info) {
    auto cm_outputs = _compiled_model.outputs();  // Output<Node>
    Napi::Array js_outputs = Napi::Array::New(info.Env(), cm_outputs.size());

    uint32_t i = 0;
    for (auto& out : cm_outputs)
        js_outputs[i++] = Output<const ov::Node>::wrap(info.Env(), out);

    return js_outputs;
}

Napi::Value CompiledModelWrap::get_input(const Napi::CallbackInfo& info) {
    try {
        return get_node(
            info,
            static_cast<const ov::Output<const ov::Node>& (ov::CompiledModel::*)() const>(&ov::CompiledModel::input),
            static_cast<const ov::Output<const ov::Node>& (ov::CompiledModel::*)(const std::string&) const>(
                &ov::CompiledModel::input),
            static_cast<const ov::Output<const ov::Node>& (ov::CompiledModel::*)(size_t) const>(
                &ov::CompiledModel::input));
    } catch (std::exception& e) {
        reportError(info.Env(), e.what() + std::string("inputs."));
        return info.Env().Null();
    }
}

Napi::Value CompiledModelWrap::get_inputs(const Napi::CallbackInfo& info) {
    auto cm_inputs = _compiled_model.inputs();  // Output<Node>
    Napi::Array js_inputs = Napi::Array::New(info.Env(), cm_inputs.size());

    uint32_t i = 0;
    for (auto& out : cm_inputs)
        js_inputs[i++] = Output<const ov::Node>::wrap(info.Env(), out);

    return js_inputs;
}

Napi::Value CompiledModelWrap::get_node(
    const Napi::CallbackInfo& info,
    const ov::Output<const ov::Node>& (ov::CompiledModel::*func)() const,
    const ov::Output<const ov::Node>& (ov::CompiledModel::*func_tname)(const std::string&) const,
    const ov::Output<const ov::Node>& (ov::CompiledModel::*func_idx)(size_t) const) {
    if (info.Length() == 0) {
        return Output<const ov::Node>::wrap(info.Env(), (_compiled_model.*func)());
    } else if (info.Length() != 1) {
        OPENVINO_THROW(std::string("Invalid number of arguments."));
    } else if (info[0].IsString()) {
        auto tensor_name = info[0].ToString();
        return Output<const ov::Node>::wrap(info.Env(), (_compiled_model.*func_tname)(tensor_name));
    } else if (info[0].IsNumber()) {
        auto idx = info[0].As<Napi::Number>().Int32Value();
        return Output<const ov::Node>::wrap(info.Env(), (_compiled_model.*func_idx)(idx));
    } else {
        OPENVINO_THROW(std::string("Error while getting compiled model "));
    }
}

Napi::Value CompiledModelWrap::export_model(const Napi::CallbackInfo& info) {
    std::stringstream _stream;
    _compiled_model.export_model(_stream);
    const auto& exported = _stream.str();
    return Napi::Buffer<const char>::Copy(info.Env(), exported.c_str(), exported.size());
}
