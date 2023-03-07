// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "cpp/ie_cnn_network.h"
#include "gna_plugin_config.hpp"
#include "openvino/core/model.hpp"

namespace ov {
namespace intel_gna {

class TransformationsPipeline {
public:
    explicit TransformationsPipeline(const Config& config) : config(config) {
        effective_compile_target = config.target->get_effective_compile_target();
    }
    void apply(const std::shared_ptr<ov::Model>& model);
    IE_SUPPRESS_DEPRECATED_START
    void apply_legacy(const InferenceEngine::CNNNetwork& network, bool runBeforeCopy);
    void convert_precision_legacy(InferenceEngine::CNNNetwork& network);
    IE_SUPPRESS_DEPRECATED_END
    bool is_fake_quantized() {
        return fake_quantized;
    };
    const ov::intel_gna::Config& config;

private:
    bool is_ngraph_passes_used = false;
    bool fake_quantized = false;
    int legacy_pass_index = 0;
    ov::intel_gna::target::DeviceVersion effective_compile_target;
};

}  // namespace intel_gna
}  // namespace ov
