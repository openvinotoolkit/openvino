#include "ie_core.h"

Napi::Object IECore::Init(Napi::Env env, Napi::Object exports) {
    Napi::Function func = DefineClass(env, "IECore", {
        InstanceMethod("getAvailableDevices", &IECore::getAvailableDevices),
        InstanceMethod("getMetric", &IECore::getMetric)
    });

    constructor = Napi::Persistent(func);
    constructor.SuppressDestruct();
    exports.Set("IECore", func);

    return exports;
}

IECore::IECore(const Napi::CallbackInfo &info) : Napi::ObjectWrap<IECore>(info) {
    this->_ie_core = std::make_shared<InferenceEngine::Core>();
}

Napi::FunctionReference IECore::constructor;

Napi::Value IECore::getAvailableDevices(const Napi::CallbackInfo &info) {
    Napi::Env env = info.Env();
    Napi::Array result = Napi::Array::New(env);
    std::vector<std::string> devices = this->_ie_core->GetAvailableDevices();

    int size = devices.size();
    for (int i = 0; i < size; i++) {
        Napi::String val = Napi::String::New(env, devices[i]);
        result[i] = val;
    }
    return result;
}


Napi::Value IECore::getMetric(const Napi::CallbackInfo &info){

    std::string device = std::string(info[0].ToString());
    std::string metric = std::string(info[1].ToString());
    InferenceEngine::Parameter parameter = this->_ie_core->GetMetric(device, metric);

    return parseParameter(info.Env(), parameter);
}

Napi::Object Init (Napi::Env env, Napi::Object exports) {
    IECore::Init(env, exports);
    return exports;
}

NODE_API_MODULE(NODE_GYP_MODULE_NAME, Init)
