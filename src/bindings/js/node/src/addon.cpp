// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <napi.h>

#include "compiled_model.hpp"
#include "core_wrap.hpp"
#include "element_type.hpp"
#include "infer_request.hpp"
#include "model_wrap.hpp"
#include "node_output.hpp"
#include "openvino/openvino.hpp"
#include "resize_algorithm.hpp"
#include "tensor.hpp"
#include "partial_shape_wrap.hpp"
#include "preprocess/preprocess_wrap.hpp"

Napi::PropertyDescriptor element = Napi::PropertyDescriptor::Accessor<enumElementType>("element");

/** @brief Initialize native add-on */
Napi::Object InitAll(Napi::Env env, Napi::Object exports) {
    ModelWrap::Init(env, exports);
    CoreWrap::Init(env, exports);
    CompiledModelWrap::Init(env, exports);
    InferRequestWrap::Init(env, exports);
    TensorWrap::Init(env, exports);
    Output<const ov::Node>::Init(env, exports);
    Output<ov::Node>::Init(env, exports);
    PartialShapeWrap::Init(env, exports);
    PreProcessWrap::Init(env, exports);

    exports.DefineProperty(element);
    
    return exports;
}

/** @brief Register and initialize native add-on */
NODE_API_MODULE(addon_openvino, InitAll)
