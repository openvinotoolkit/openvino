#include "infer_request.hpp"

#include "node_output.hpp"
#include "tensor.hpp"

InferRequestWrap::InferRequestWrap(const Napi::CallbackInfo& info) : Napi::ObjectWrap<InferRequestWrap>(info) {}

Napi::Function InferRequestWrap::GetClassConstructor(Napi::Env env) {
    return DefineClass(env,
                       "InferRequest",
                       {InstanceMethod("set_input_tensor", &InferRequestWrap::set_input_tensor),
                        InstanceMethod("infer", &InferRequestWrap::infer),
                        InstanceMethod("getTensor", &InferRequestWrap::get_tensor),
                        InstanceMethod("getOutputTensors", &InferRequestWrap::get_output_tensors),
                        InstanceMethod("get_output_tensor", &InferRequestWrap::get_output_tensor)});
}

Napi::Object InferRequestWrap::Init(Napi::Env env, Napi::Object exports) {
    auto func = GetClassConstructor(env);

    Napi::FunctionReference* constructor = new Napi::FunctionReference();
    *constructor = Napi::Persistent(func);
    env.SetInstanceData(constructor);

    exports.Set("InferRequest", func);
    return exports;
}

void InferRequestWrap::set_infer_request(const ov::InferRequest& infer_request) {
    _infer_request = infer_request;
}

Napi::Object InferRequestWrap::Wrap(Napi::Env env, ov::InferRequest infer_request) {
    Napi::HandleScope scope(env);
    Napi::Object obj = GetClassConstructor(env).New({});
    InferRequestWrap* ir = Napi::ObjectWrap<InferRequestWrap>::Unwrap(obj);
    ir->set_infer_request(infer_request);
    return obj;
}

Napi::Value InferRequestWrap::set_input_tensor(const Napi::CallbackInfo& info) {
    auto* tensorWrap = Napi::ObjectWrap<TensorWrap>::Unwrap(info[0].ToObject());
    ov::Tensor t = tensorWrap->get_tensor();

    _infer_request.set_input_tensor(t);
    return Napi::Value();
}

Napi::Value InferRequestWrap::infer(const Napi::CallbackInfo& info) {
    if (info.Length() == 0)
        _infer_request.infer();
    else if (info.Length() == 1) {
        const auto elem = info[0];

        if (elem.IsArray()) {
            auto array = elem.As<Napi::Array>();
            size_t i = 0;

            // TO_DO
            Napi::Value arrayItem = array[i];
            if (!arrayItem.IsObject()) {
                throw std::invalid_argument(std::string("Passed array must contain objects."));
            }
            Napi::Object obj = arrayItem.As<Napi::Object>();

            auto* tensorWrap = Napi::ObjectWrap<TensorWrap>::Unwrap(obj);
            ov::Tensor tensor = tensorWrap->get_tensor();

            _infer_request.set_input_tensor(tensor);
            _infer_request.infer();

        } else if (elem.IsObject()) {
            // elem z js = {"input:0": tensor}
            Napi::Object obj = elem.As<Napi::Object>();

            // for (const auto& it : obj) {
            //     if(input.IsString()){
            //         std::cout<< (it.first).As<String>().Utf8Value()<< " ";
            //     }
            // }

            auto keys = obj.GetPropertyNames();
            for (size_t i = 0; i < keys.Length(); i++) {
                auto tensor_name = static_cast<Napi::Value>(keys[i]).ToString().Utf8Value();
                auto tensor_obj = obj.Get(tensor_name).As<Napi::Object>();

                auto* tensorWrap = Napi::ObjectWrap<TensorWrap>::Unwrap(tensor_obj);
                ov::Tensor t = tensorWrap->get_tensor();

                _infer_request.set_tensor(tensor_name, t);
            }
            _infer_request.infer();
        }
    } else {
        reportError(info.Env(), "Inference error.");
    }
    return Napi::Value();
}

Napi::Value InferRequestWrap::get_tensor(const Napi::CallbackInfo& info) {
    auto* outputWrap = Napi::ObjectWrap<Output<const ov::Node>>::Unwrap(info[0].ToObject());
    ov::Output<const ov::Node> output = outputWrap->get_output();

    ov::Tensor tensor = _infer_request.get_tensor(output);
    return TensorWrap::Wrap(info.Env(), tensor);
}

Napi::Value InferRequestWrap::get_output_tensor(const Napi::CallbackInfo& info) {
    ov::Tensor tensor = _infer_request.get_output_tensor();
    return TensorWrap::Wrap(info.Env(), tensor);
}

Napi::Value InferRequestWrap::get_output_tensors(const Napi::CallbackInfo& info) {
    auto compiled_model = _infer_request.get_compiled_model().outputs();
    auto outputs_obj = Napi::Object::New(info.Env());

    for (auto& node : compiled_model) {
        auto tensor = _infer_request.get_tensor(node);
        outputs_obj.Set(node.get_any_name(), TensorWrap::Wrap(info.Env(), tensor));
    }

    return outputs_obj;
}
