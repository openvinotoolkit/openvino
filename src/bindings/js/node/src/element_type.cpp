#include "element_type.hpp"

#include <iostream>
#include <typeinfo>

Napi::Value enumElementType(const Napi::CallbackInfo& info) {
    Napi::Object enumObj = Napi::Object::New(info.Env());
    std::vector<Napi::PropertyDescriptor> pds;

    for (auto it = element_type_map.begin(); it != element_type_map.end(); it++)
        pds.push_back(
            Napi::PropertyDescriptor::Value(it->first, Napi::String::New(info.Env(), it->first), napi_default));

    enumObj.DefineProperties(pds);
    return enumObj;
}
