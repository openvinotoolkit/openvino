#include "session.hpp"

#include "openvino/openvino.hpp"
#include "shape_lite.hpp"
#include "tensor.hpp"

std::shared_ptr<ov::Model> loadModel(std::string xml_path, std::string bin_path) {
    ov::Core core;
    try {
        return core.read_model(xml_path, bin_path);
    } catch (const std::exception& e) {
        std::cout << "== Error in load_model: " << e.what() << std::endl;
        throw e;
    }
}

ov::CompiledModel compileModel(std::shared_ptr<ov::Model> model, ov::Shape shape, std::string layout) {
    ov::Layout tensor_layout = ov::Layout(layout);

    ov::Core core;
    std::cout << "== Model name: " << model->get_friendly_name() << std::endl;

    ov::element::Type input_type = ov::element::u8;
    ov::preprocess::PrePostProcessor ppp(model);
    ppp.input().tensor().set_shape(shape).set_element_type(input_type).set_layout(tensor_layout);
    ppp.input().preprocess().resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR);
    ppp.output().tensor().set_element_type(ov::element::f32);
    ppp.input().model().set_layout(tensor_layout);
    ppp.build();
    ov::CompiledModel compiled_model;
    const std::string backend = "TEMPLATE";
    try {
        compiled_model = core.compile_model(model, backend);
    } catch (const std::exception& e) {
        std::cout << "== Error in compile_model: " << e.what() << std::endl;
        throw e;
    }

    return compiled_model;
}

ov::Tensor performInference(ov::CompiledModel cm, ov::Tensor t) {
    ov::InferRequest infer_request = cm.create_infer_request();
    infer_request.set_input_tensor(t);
    infer_request.infer();

    return infer_request.get_output_tensor();
}

Session::Session(const Napi::CallbackInfo& info) : Napi::ObjectWrap<Session>(info) {
    auto xml_path = info[0].ToString();
    auto bin_path = info[1].ToString();
    Napi::Object obj = info[2].ToObject();
    auto* shape = Napi::ObjectWrap<ShapeLite>::Unwrap(obj);
    auto layout = info[3].ToString();

    auto model = loadModel(xml_path, bin_path);
    try {
        this->_cmodel = compileModel(model, shape->get_original(), layout);
    } catch (const std::exception& e) {
        std::cout << "== Error in Session constructor: " << e.what() << std::endl;
        throw e;
    }
};

Napi::Function Session::GetClassConstructor(Napi::Env env) {
    return DefineClass(env, "Session", {InstanceMethod("infer", &Session::infer)});
};

Napi::Object Session::Init(Napi::Env env, Napi::Object exports) {
    auto func = GetClassConstructor(env);

    Napi::FunctionReference* constructor = new Napi::FunctionReference();
    *constructor = Napi::Persistent(func);
    env.SetInstanceData(constructor);

    exports.Set("Session", func);
    return exports;
};

Napi::Value Session::infer(const Napi::CallbackInfo& info) {
    std::cout << "== Run inference" << std::endl;
    ov::Tensor output_tensor;
    Napi::Object obj = info[0].ToObject();
    auto* input_tensor = Napi::ObjectWrap<TensorWrap>::Unwrap(obj);

    try {
        output_tensor = performInference(this->_cmodel, input_tensor->get_tensor());
    } catch (const std::exception& e) {
        std::cout << "== Error in run: " << e.what() << std::endl;
        throw e;
    }

    return TensorWrap::Wrap(info.Env(), output_tensor);
};
