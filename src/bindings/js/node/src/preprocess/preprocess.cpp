// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "preprocess/preprocess.hpp"

#include "addon.hpp"

namespace preprocess {
  Napi::Object init(Napi::Env env, Napi::Object exports) {
      auto preprocess = Napi::PropertyDescriptor::Accessor<add_preprocess_namespace>("preprocess");

      exports.DefineProperty(preprocess);

      return exports;
  }

  Napi::Value add_preprocess_namespace(const Napi::CallbackInfo& info) {
      Napi::Env env = info.Env();
      auto preprocess = Napi::Object::New(env);
      auto resizeAlgorithm = Napi::PropertyDescriptor::Accessor<enumResizeAlgorithm>("resizeAlgorithm");

      const auto data = env.GetInstanceData<AddonData>();
      init_class(env, preprocess, "PrePostProcessor", &PrePostProcessorWrap::get_class, data->ppp);
      preprocess.DefineProperty(resizeAlgorithm);

      return preprocess;
  }
};
