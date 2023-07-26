#include "resize_algorithm.hpp"

#include <iostream>
#include <typeinfo>

#include <openvino/runtime/core.hpp>

Napi::Value enumResizeAlgorithm(const Napi::CallbackInfo& info) {
    Napi::Object enumObj = Napi::Object::New(info.Env());
    std::vector<Napi::PropertyDescriptor> pds;

    std::string resizeAlgorithms[] = {
      "RESIZE_LINEAR",
      "RESIZE_CUBIC",
      "RESIZE_NEAREST"
    };

    for (std::string& algorithm: resizeAlgorithms) {
      pds.push_back(
        Napi::PropertyDescriptor::Value(
          algorithm,
          Napi::String::New(info.Env(), algorithm),
          napi_default
        )
      );
    }

    enumObj.DefineProperties(pds);
    return enumObj;
}
