// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "node/include/addon.hpp"

#include <napi.h>

#include "node/include/compiled_model.hpp"
#include "node/include/core_wrap.hpp"
#include "node/include/element_type.hpp"
#include "node/include/errors.hpp"
#include "node/include/helper.hpp"
#include "node/include/infer_request.hpp"
#include "node/include/model_wrap.hpp"
#include "node/include/node_output.hpp"
#include "node/include/partial_shape_wrap.hpp"
#include "node/include/preprocess/preprocess.hpp"
#include "node/include/tensor.hpp"
#include "node/include/type_validation.hpp"
#include "openvino/openvino.hpp"

void init_class(Napi::Env env,
                Napi::Object exports,
                std::string class_name,
                Prototype func,
                Napi::FunctionReference& reference) {
    const auto& prototype = func(env);

    reference = Napi::Persistent(prototype);
    exports.Set(class_name, prototype);
}

Napi::Value save_model_sync(const Napi::CallbackInfo& info) {
    std::vector<std::string> allowed_signatures;
    try {
        if (ov::js::validate<ModelWrap, Napi::String>(info, allowed_signatures)) {
            const auto& model = info[0].ToObject();
            const auto m = Napi::ObjectWrap<ModelWrap>::Unwrap(model);
            const auto path = js_to_cpp<std::string>(info, 1);
            ov::save_model(m->get_model(), path);
        } else if (ov::js::validate<ModelWrap, Napi::String, Napi::Boolean>(info, allowed_signatures)) {
            const auto& model = info[0].ToObject();
            const auto m = Napi::ObjectWrap<ModelWrap>::Unwrap(model);
            const auto path = js_to_cpp<std::string>(info, 1);
            const auto compress_to_fp16 = info[2].ToBoolean();
            ov::save_model(m->get_model(), path, compress_to_fp16);
        } else {
            OPENVINO_THROW("'saveModelSync'", ov::js::get_parameters_error_msg(info, allowed_signatures));
        }
    } catch (const std::exception& e) {
        reportError(info.Env(), e.what());
    }

    return info.Env().Undefined();
}

/** @brief Initialize native add-on */
Napi::Object init_module(Napi::Env env, Napi::Object exports) {
    auto addon_data = new AddonData();
    env.SetInstanceData<AddonData>(addon_data);

    init_class(env, exports, "Model", &ModelWrap::get_class, addon_data->model);
    init_class(env, exports, "Core", &CoreWrap::get_class, addon_data->core);
    init_class(env, exports, "CompiledModel", &CompiledModelWrap::get_class, addon_data->compiled_model);
    init_class(env, exports, "InferRequest", &InferRequestWrap::get_class, addon_data->infer_request);
    init_class(env, exports, "Tensor", &TensorWrap::get_class, addon_data->tensor);
    init_class(env, exports, "Output", &Output<ov::Node>::get_class, addon_data->output);
    init_class(env, exports, "ConstOutput", &Output<const ov::Node>::get_class, addon_data->const_output);
    init_class(env, exports, "PartialShape", &PartialShapeWrap::get_class, addon_data->partial_shape);

    init_function(env, exports, "saveModelSync", save_model_sync);

    preprocess::init(env, exports);
    element::init(env, exports);

    return exports;
}

/** @brief Register and initialize native add-on */
NODE_API_MODULE(addon_openvino, init_module)
