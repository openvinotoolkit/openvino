// https://github.com/nodejs/node-addon-examples/issues/85

// examples using deferred

#include "AsyncWorkerCallback.hpp"

#include <chrono>
#include <thread>

class CallbackWorker : public Napi::AsyncWorker {
public:
    CallbackWorker(Napi::Function& callback, std::string model_onnx, Napi::Env env)
        : Napi::AsyncWorker(callback),
          model_onnx(model_onnx),
          env(env) {}
    virtual ~CallbackWorker() = default;

    // Executed inside the worker-thread.
    // It is not safe to access JS engine data structure
    // here, so everything we need for input and output
    // should go on `this`.
    void Execute() override {
        ov::Core _core;
        model = _core.read_model(model_onnx);
        std::chrono::milliseconds timespan(5000);  // or whatever

        estimate = 5;
        std::cout << "cpp " << model->get_name() << " wczytany na innym threadzie, teraz poczeka 5s\n";
        std::this_thread::sleep_for(timespan);

        // modelWrap = ModelWrap::Wrap(Env(), model);
    }

    // Executed when the async work is complete
    // this function will be run inside the main event loop
    // so it is safe to use JS engine data again
    void OnOK() override {
        Napi::HandleScope scope(env);
        // Callback().Call({Env().Undefined(), Napi::String::New(Env(),"aksjhdakjshdkja")});

        // deferred.Resolve(Napi::Number::New(Env(), estimate));

        Napi::Object obj2 = ModelWrap::GetClassConstructor(Env()).New({});
        ModelWrap* m = Napi::ObjectWrap<ModelWrap>::Unwrap(obj2);
        m->set_model(model);

        mw = ModelWrap::Wrap(Env(), model);
        // Callback().Call({env.Global(), mw });
        Callback().Call({Env().Undefined(), obj2});
    }

private:
    std::string model_onnx;
    Napi::Object mw;
    std::shared_ptr<ov::Model> model;
    Napi::Env env;
    double estimate;
};

// tylko onnx model
Napi::Value read_model_callback(const Napi::CallbackInfo& info) {
    std::string model_onnx = info[0].ToString();
    Napi::Function callback = info[1].As<Napi::Function>();
    CallbackWorker* callbackWorker = new CallbackWorker(callback, model_onnx, info.Env());
    callbackWorker->Queue();
    // return deferred.Promise();

    return info.Env().Undefined();
}
