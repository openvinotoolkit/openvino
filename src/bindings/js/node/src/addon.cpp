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

/** @brief Initialize native add-on */
Napi::Object init_all(Napi::Env env, Napi::Object exports) {
    auto addon_data = new AddonData();
    env.SetInstanceData<AddonData>(addon_data);

    ModelWrap::init(env, exports);
    CoreWrap::init(env, exports);
    CompiledModelWrap::init(env, exports);
    InferRequestWrap::init(env, exports);
    TensorWrap::init(env, exports);
    Output<const ov::Node>::init(env, exports);
    Output<ov::Node>::init(env, exports);
    PartialShapeWrap::init(env, exports);
    
    preprocess::init(env, exports);
    element::init(env, exports);
    
    return exports;
}

/** @brief Register and initialize native add-on */
NODE_API_MODULE(addon_openvino, init_all)
