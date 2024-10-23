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

namespace ov {
namespace intel_cpu {

class Transformations {
public:
    Transformations(const std::shared_ptr<ov::Model>& initialModel,
                    const Config&                     config)
        : model(initialModel),
          config(config) {
            CPU_DEBUG_CAPS_MAYBE_UNUSED(this->config);
          }

    void UpToLpt();
    void CpuSpecificOpSet();
    void PostLpt();
    void Snippets(void);

private:
    std::shared_ptr<ov::Model> model;
    const Config& config;

    void PreLpt(const std::vector<ov::element::Type>& defaultPrecisions);

    void Lpt(const std::vector<ov::element::Type>& defaultPrecisions);

    void MainSnippets(void);

    void PostSnippets(void);

    bool is_decompression_multiply(const std::shared_ptr<const ov::Node>& node) const;

    static bool fuse_type_to_convert(const std::shared_ptr<ov::Node>& node, const precisions_map& precisions);
    static bool fuse_type_to_fq(const std::shared_ptr<ov::Node>& node, const precisions_map& precisions);
    static bool fuse_type_to_pa(const std::shared_ptr<ov::Node>& node, const precisions_map& precisions);
};

}   // namespace intel_cpu
}   // namespace ov
