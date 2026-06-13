// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <optional>
#include <string>

#include "intel_npu/npu_private_properties.hpp"
#include "intel_npu/utils/zero/zero_init.hpp"
#include "model_serializer.hpp"

namespace ov::test::behavior {

inline ::intel_npu::SerializedIR makeTestSerializedIR(const std::shared_ptr<ov::Model>& model,
                                                      const std::shared_ptr<::intel_npu::ZeroInitStructsHolder>& init) {
    // The test is not concerned with validating the serialization algorithm. Choose the "all-weights-copy" as the
    // safest version.
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
