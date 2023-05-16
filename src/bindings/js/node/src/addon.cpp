#include <napi.h>

#include "compiled_model.hpp"
#include "core_wrap.hpp"
#include "element_type.hpp"
#include "infer_request.hpp"
#include "model_wrap.hpp"
#include "openvino/openvino.hpp"
#include "pre_post_process_wrap.hpp"
#include "shape_lite.hpp"
#include "tensor.hpp"

Napi::String Method(const Napi::CallbackInfo& info) {
    ov::Version version = ov::get_openvino_version();
    std::string str;
    return Napi::String::New(info.Env(), str.assign(version.buildNumber));
}

/// @brief Initialize native add-on
Napi::Object InitAll(Napi::Env env, Napi::Object exports) {
    ModelWrap::Init(env, exports);
    CoreWrap::Init(env, exports);
    CompiledModelWrap::Init(env, exports);
    InferRequestWrap::Init(env, exports);
    TensorWrap::Init(env, exports);
    ShapeLite::Init(env, exports);
    PrePostProcessorWrap::Init(env, exports);
    Napi::PropertyDescriptor element = Napi::PropertyDescriptor::Accessor<enumElementType>("element");
    exports.DefineProperty(element);

    exports.Set(Napi::String::New(env, "getDescriptionString"), Napi::Function::New(env, Method));

    return exports;
}

/// @brief Register and initialize native add-on
NODE_API_MODULE(addon_openvino, InitAll)
