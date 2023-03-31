// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/model.hpp"
#include "utils/debug_capabilities.h"
#include "low_precision/low_precision.hpp"
#include "config.h"
#include "transformations/convert_precision.hpp"

#include "itt.h"

#include <memory>
#include <vector>

using namespace InferenceEngine;

#define IE_CPU_PLUGIN_THROW(...) IE_THROW(__VA_ARGS__) << "CPU plugin: "

namespace ov {
namespace intel_cpu {

class Transformations {
public:
    Transformations(const std::shared_ptr<ov::Model>& initialModel,
                    const bool                        enableLpt,
                    const bool                        enableBF16,
                    const bool                        isLegacyApi,
                    const Config::SnippetsMode&       snippetsMode,
                    const Config&                     config)
        : model(initialModel),
          enableLpt(enableLpt),
          enableBF16(enableBF16),
          isLegacyApi(isLegacyApi),
          snippetsMode(snippetsMode),
          config(config) {
            CPU_DEBUG_CAPS_MAYBE_UNUSED(this->config);
          }

    void UpToCpuSpecificOpSet();
    void CpuSpecificOpSet();

private:
    std::shared_ptr<ov::Model> model;
    const bool    enableLpt;
    const bool    enableBF16;
    const bool    isLegacyApi;
    const Config::SnippetsMode snippetsMode;
    const Config& config;

    void PreLpt(const std::vector<ov::element::Type>& defaultPrecisions, const bool isLegacyApi);

    void Lpt(const bool hasINT16orINT32Levels, const std::vector<ov::element::Type>& defaultPrecisions);

    void PostLpt();

    void MainSnippets(void);

    void PostSnippets(void);

    void Snippets(void);

    static bool fuse_type_to_convert(const std::shared_ptr<ngraph::Node>& node, const precisions_map& precisions);
};

}   // namespace intel_cpu
}   // namespace ov
