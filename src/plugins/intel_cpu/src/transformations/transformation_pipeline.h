// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "config.h"
#include "openvino/core/model.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"
#include "transformations/convert_precision.hpp"

namespace ov::intel_cpu {

class Transformations {
public:
    Transformations(std::shared_ptr<ov::Model> initialModel, const Config& config)
        : model(std::move(initialModel)),
          config(config) {}

    void UpToLpt();
    void CpuSpecificOpSet();
    void PostLpt();
    void Snippets();

private:
    std::shared_ptr<ov::Model> model;
    const Config& config;

    void PreLpt(const std::vector<ov::element::Type>& defaultPrecisions);

    void Lpt(const std::vector<ov::element::Type>& defaultPrecisions);
    void runLptPasses(const std::vector<ov::element::Type>& defaultPrecisions);

    void MainSnippets();

    void PostSnippets();

    static bool is_decompression_multiply(const std::shared_ptr<const ov::Node>& node);

    static bool fuse_type_to_convert(const std::shared_ptr<ov::Node>& node, const precisions_map& precisions);
    static bool fuse_type_to_fq(const std::shared_ptr<ov::Node>& node, const precisions_map& precisions);
    static bool fuse_type_to_pa(const std::shared_ptr<ov::Node>& node, const precisions_map& precisions);
};

}  // namespace ov::intel_cpu
