#include "ModelWrap.hpp"

ModelWrap::ModelWrap(const Napi::CallbackInfo& info) : Napi::ObjectWrap<ModelWrap>(info) {}

Napi::Function ModelWrap::GetClassConstructor(Napi::Env env) {
    return DefineClass(env,
                       "ModelWrap",
                       {InstanceMethod("get_name", &ModelWrap::get_name),
                        InstanceMethod("read_model", &ModelWrap::read_model),
                        InstanceMethod("compile", &ModelWrap::compile_model),
                        InstanceMethod("infer", &ModelWrap::infer)});
}

Napi::Object ModelWrap::Init(Napi::Env env, Napi::Object exports) {
    Napi::Function func = GetClassConstructor(env);

    Napi::FunctionReference* constructor = new Napi::FunctionReference();
    *constructor = Napi::Persistent(func);
    env.SetInstanceData(constructor);

    exports.Set("Model", func);
    return exports;
}

void ModelWrap::set_model(const std::shared_ptr<ov::Model>& model) {
    _model = model;
}

Napi::Object ModelWrap::Wrap(Napi::Env env, std::shared_ptr<ov::Model> model) {
    Napi::HandleScope scope(env);
    Napi::Object obj = ModelWrap::GetClassConstructor(env).New({});
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

Napi::Value ModelWrap::read_model(const Napi::CallbackInfo& info) {
    if (info.Length() == 1 && info[0].IsString()) {
        std::string model_onnx = info[0].ToString();
        std::shared_ptr<ov::Model> model = _core.read_model(model_onnx);
        return ModelWrap::Wrap(info.Env(), model);
    } else if (info.Length() != 2) {
        reportError(info.Env(), "Invalid number of arguments -> " + std::to_string(info.Length()));
        return Napi::Value();
    } else if (info[0].IsString() && info[1].IsString()) {
        std::string model_path = info[0].ToString();
        std::string bin_path = info[1].ToString();
        std::shared_ptr<ov::Model> model = _core.read_model(model_path, bin_path);
        return ModelWrap::Wrap(info.Env(), model);
    } else {
        reportError(info.Env(), "Error while reading model.");
        return Napi::Value();
    }
}

Napi::Value ModelWrap::compile_model(const Napi::CallbackInfo& info) {
    if (info.Length() != 1) {
        reportError(info.Env(), "Invalid number of arguments -> " + std::to_string(info.Length()));
        return Napi::Value();
    } else if (info[0].IsString()) {
        std::string device = info[0].ToString();
        _compiled_model = _core.compile_model(_model, device);
        return info.This();
    } else {
        reportError(info.Env(), "Error while compiling model.");
        return Napi::Value();
    }
}

Napi::Value ModelWrap::infer(const Napi::CallbackInfo& info) {
    ov::InferRequest _infer_request = _compiled_model.create_infer_request();

    auto* tensorWrap = Napi::ObjectWrap<TensorWrap>::Unwrap(info[0].ToObject());
    ov::Tensor t = tensorWrap->get_tensor();
    _infer_request.set_input_tensor(t);

    _infer_request.infer();
    ov::Tensor tensor = _infer_request.get_output_tensor();
    return TensorWrap::Wrap(info.Env(), tensor);
}
