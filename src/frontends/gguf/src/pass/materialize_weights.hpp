// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <vector>

#include "openvino/pass/pass.hpp"
#include "quant/gguf.hpp"

namespace ov::frontend::gguf::pass {

class MaterializeGGUFWeights : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::frontend::gguf::pass::MaterializeGGUFWeights");
    struct SourceRows {
        GgufWeight source;
        size_t first;
        size_t count;
    };
    using Materializer = std::function<std::shared_ptr<ov::Node>(const std::shared_ptr<ov::Node>&,
                                                                 const std::vector<SourceRows>&,
                                                                 const std::vector<int64_t>&)>;

    explicit MaterializeGGUFWeights(Materializer materializer = {}) : m_materializer(std::move(materializer)) {}
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

private:
    Materializer m_materializer;
};

}  // namespace ov::frontend::gguf::pass
