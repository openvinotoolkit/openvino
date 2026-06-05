// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <string>

#include "intel_npu/utils/zero/zero_init.hpp"
#include "model_serializer.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"

namespace ov::test::behavior {

inline ::intel_npu::SerializedIR makeTestSerializedIR(std::shared_ptr<ov::Model> model,
                                                      std::shared_ptr<::intel_npu::ZeroInitStructsHolder> init) {
    auto props = init->getCompilerProperties();
    return ::intel_npu::compiler_utils::serializeIR(model,
                                                    props.compilerVersion,
                                                    props.maxOVOpsetVersionSupported,
                                                    ov::intel_npu::ModelSerializerVersion::ALL_WEIGHTS_COPY,
                                                    [](const std::string&, const std::optional<std::string>&) {
                                                        return true;
                                                    });
}

}  // namespace ov::test::behavior
