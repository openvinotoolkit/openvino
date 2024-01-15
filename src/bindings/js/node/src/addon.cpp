// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "addon.hpp"

#include <napi.h>

#include "compiled_model.hpp"
#include "core_wrap.hpp"
#include "element_type.hpp"
#include "infer_request.hpp"
#include "model_wrap.hpp"
#include "node_output.hpp"
#include "openvino/openvino.hpp"
#include "partial_shape_wrap.hpp"
#include "preprocess/preprocess.hpp"
#include "tensor.hpp"

void init_class(Napi::Env env,
                Napi::Object exports,
                std::string class_name,
                Prototype func,
                Napi::FunctionReference& reference) {
    const auto& prototype = func(env);

    reference = Napi::Persistent(prototype);
    exports.Set(class_name, prototype);
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

    preprocess::init(env, exports);
    element::init(env, exports);

    return exports;
}

/** @brief Register and initialize native add-on */
NODE_API_MODULE(addon_openvino, init_module)
