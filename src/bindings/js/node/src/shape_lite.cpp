#include "shape_lite.hpp"

ShapeLite::ShapeLite(const Napi::CallbackInfo& info) : Napi::ObjectWrap<ShapeLite>(info) {
    if (info.Length() != 2 && info.Length() != 0)  // default contructor takes 0 args
        reportError(info.Env(), "Invalid number of arguments for ShapeLite constructor.");
    else if (info.Length() == 2) {
        auto dim = info[0].As<Napi::Number>().Int32Value();
        auto data_array = info[1].As<Napi::Uint32Array>();
        for (int i = 0; i < dim; i++)
            this->_shape.push_back(data_array[i]);
    }
}

Napi::Function ShapeLite::GetClassConstructor(Napi::Env env) {
    return DefineClass(env,
                       "ShapeLite",
                       {InstanceAccessor<&ShapeLite::get_data>("data"),
                        InstanceMethod("getDim", &ShapeLite::get_dim),
                        InstanceMethod("shapeSize", &ShapeLite::shape_size),
                        InstanceMethod("getData", &ShapeLite::get_data)});
}

Napi::Object ShapeLite::Init(Napi::Env env, Napi::Object exports) {
    auto func = GetClassConstructor(env);

    Napi::FunctionReference* constructor = new Napi::FunctionReference();
    *constructor = Napi::Persistent(func);
    env.SetInstanceData(constructor);

    exports.Set("Shape", func);
    return exports;
}

Napi::Object ShapeLite::Wrap(Napi::Env env, ov::Shape shape) {
    Napi::HandleScope scope(env);
    auto obj = GetClassConstructor(env).New({});
    auto t = Napi::ObjectWrap<ShapeLite>::Unwrap(obj);
    t->set_shape(shape);
    return obj;
}

Napi::Value ShapeLite::get_data(const Napi::CallbackInfo& info) {
    auto arr = Napi::Uint32Array::New(info.Env(), _shape.size());
    for (size_t i = 0; i < _shape.size(); i++)
        arr[i] = _shape[i];

    return arr;
}

Napi::Value ShapeLite::get_dim(const Napi::CallbackInfo& info) {
    return Napi::Number::New(info.Env(), this->_shape.size());
}

Napi::Value ShapeLite::shape_size(const Napi::CallbackInfo& info) {
    return Napi::Number::New(info.Env(), ov::shape_size(this->_shape));
}

void ShapeLite::set_shape(const ov::Shape& shape) {
    _shape = shape;
}

ov::Shape ShapeLite::get_original() {
    return this->_shape;
}
