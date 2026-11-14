// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/core/any.hpp"
#include "openvino/core/extension.hpp"
#include "openvino/frontend/gguf/visibility.hpp"

namespace ov::frontend::gguf {

class SharedWeights;

/// Share the weights of a GgufDecoder between the models of several convert() calls.
///
/// A caller that converts the same weights more than once (the llama.cpp cgraph path compiles one
/// model per graph topology) registers one instance on every FrontEnd it converts with. The first
/// conversion of a weight keeps its extracted tensors in page-aligned host buffers; later
/// conversions reuse them instead of extracting again. Compiling with get_compile_properties()
/// lets a plugin import those buffers instead of copying them, so the compiled models hold one
/// copy of the weights between them.
///
/// Weights are identified by their GgufDecoder data pointer, so an instance must not outlive the
/// weights it was used with. The GPU plugin imports the buffers on any GPU; register the extension
/// only for devices that read host memory directly (integrated GPUs).
class GGUF_FRONTEND_API WeightSharingExtension : public ov::Extension {
public:
    OPENVINO_RTTI("gguf::WeightSharingExtension", "", ov::Extension);

    WeightSharingExtension();
    ~WeightSharingExtension() override;

    /// Compile properties that let a plugin import the shared weights; pass them to compile_model()
    /// for every model converted with this extension. They cover the weights converted so far, so
    /// take them after converting the model they are compiled with.
    ov::AnyMap get_compile_properties() const;

private:
    friend class FrontEnd;
    std::shared_ptr<SharedWeights> m_weights;
};

}  // namespace ov::frontend::gguf
