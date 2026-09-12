// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>

#include "openvino/frontend/gguf/builder/metadata.hpp"
#include "openvino/frontend/gguf/visibility.hpp"

namespace ov {
namespace frontend {
namespace gguf {

// Opaque graph produced by GgufGraphContext; defined in builder/gguf_graph.hpp.
struct GgufGraph;

namespace detail {
struct WeightStore;
}

// Borrowed metadata and weights, valid only during the synchronous factory/build call.
// Returned graphs retain the weight storage; builders must not retain this view afterward.
struct GGUF_FRONTEND_API BuildContext {
    GgufMetadata metadata;

    // `general.architecture` as the file spells it.
    std::string arch;

    // Opaque weight tables accessed through GgufTensors.
    detail::WeightStore* weights = nullptr;
};

// Whole-model builder for any family, registered through ArchitectureDefinition.
// See docs/porting_a_llama_cpp_model.md for external and built-in registration.
class GGUF_FRONTEND_API ModelBuilder {
public:
    virtual ~ModelBuilder();

    // Emit the whole model and return the finished graph.
    virtual std::shared_ptr<GgufGraph> build() = 0;
};

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
