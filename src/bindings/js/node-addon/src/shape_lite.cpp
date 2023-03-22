#include "shape_lite.hpp"

ShapeLite::ShapeLite(const Napi::CallbackInfo& info) {
    if (info.Length() != 2 && info.Length() != 0)  // default contructor takes 0 args
        reportError(info.Env(), "Invalid number of arguments for ShapeLite constructor.");
    else if (info.Length() == 2) {
        auto dim = info[0].As<Napi::Number>();
        auto data_array = info[1].As<Napi::Uint32Array>();
        for (size_t i = 0; i < dim; i++)
            this->shape.push_back(data_array[i]);
    }
}

Napi::Uint32Array ShapeLite::get_data() {
    auto shape = _tensor.get_shape();
    auto arr = Napi::Uint32Array::New(info.Env(), shape.size());
    for (size_t i = 0; i < shape.size(); i++)
        arr[i] = shape[i];

    return arr;
}

Napi::Number ShapeLite::get_dim(const Napi::CallbackInfo& info) {
    return Napi::Number::New(info.Env(), this->shape.size());
}

Napi::Number ShapeLite::shape_size(const Napi::CallbackInfo& info) {
    return Napi::Number::New(info.Env(), ov::shape_size(this->shape));
}
