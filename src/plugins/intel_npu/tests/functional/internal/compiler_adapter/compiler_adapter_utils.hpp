// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <string>

#include "intel_npu/common/filtered_config.hpp"
#include "intel_npu/config/options.hpp"
#include "intel_npu/utils/zero/zero_init.hpp"
#include "model_serializer.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"

namespace ov::test::behavior {

using namespace ::intel_npu;

inline SerializedIR makeTestSerializedIR(const std::shared_ptr<ov::Model>& model,
                                         const std::shared_ptr<ZeroInitStructsHolder>& init) {
    auto props = init->getCompilerProperties();
    return compiler_utils::serializeIR(model,
                                       props.compilerVersion,
                                       props.maxOVOpsetVersionSupported,
                                       ov::intel_npu::ModelSerializerVersion::ALL_WEIGHTS_COPY,
                                       [](const std::string&, const std::optional<std::string>&) {
                                           return true;
                                       });
}

inline FilteredConfig makeTestCompileConfig() {
    auto options = std::make_shared<OptionsDesc>();
    options->add<LOG_LEVEL>();
    options->add<MODEL_SERIALIZER_VERSION>();
    options->add<CACHE_DIR>();
    options->add<BYPASS_UMD_CACHING>();
    options->add<CACHE_ENCRYPTION_CALLBACKS>();
    options->add<CREATE_EXECUTOR>();
    options->add<DEFER_WEIGHTS_LOAD>();

    FilteredConfig config(options);
    config.enable(LOG_LEVEL::key().data(), true);
    config.enable(MODEL_SERIALIZER_VERSION::key().data(), true);
    config.enable(CACHE_DIR::key().data(), true);
    config.enable(BYPASS_UMD_CACHING::key().data(), true);
    config.enable(CACHE_ENCRYPTION_CALLBACKS::key().data(), true);
    config.enable(CREATE_EXECUTOR::key().data(), true);
    config.enable(DEFER_WEIGHTS_LOAD::key().data(), true);

    Config::ConfigMap values;
    values[BYPASS_UMD_CACHING::key().data()] = BYPASS_UMD_CACHING::toString(true);
    values[CREATE_EXECUTOR::key().data()] = CREATE_EXECUTOR::toString(0);
    config.update(values);

    return config;
}

}  // namespace ov::test::behavior
