// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

#include "openvino/core/any.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/weight_sharing_util.hpp"
#include "quant/gguf.hpp"

namespace ov::frontend::gguf {

// Weight subgraphs kept across conversions for WeightSharingExtension. Every constant of a kept
// subgraph lives in its own page-aligned, page-padded host buffer registered in the weight-sharing
// context: the GPU plugin asserts cacheline alignment of an imported buffer and maps whole pages.
class SharedWeights {
public:
    // A fresh copy of the subgraph kept for `source`, built by `build` on first use. Copies share
    // the constants' buffers, so every conversion gets its own nodes over the same weights.
    Output<Node> get(const GgufWeight& source, const std::function<Output<Node>()>& build);

    // Carries a copy of the context: a plugin reads it without this lock while later conversions
    // register more weights, so it covers the weights converted so far.
    ov::AnyMap get_compile_properties() const;

private:
    // Decoder data pointer, byte size, GGUF type, logical shape and weight name.
    using Key = std::tuple<const void*, size_t, GgufTensorType, ov::Shape, std::string>;

    std::shared_ptr<Node> make_shared_constant(const ov::op::v0::Constant& constant);

    mutable std::mutex m_mutex;
    const std::shared_ptr<ov::weight_sharing::Context> m_context = std::make_shared<ov::weight_sharing::Context>();
    std::map<Key, Output<Node>> m_weights;
    // The context only observes its sources; the buffers live as long as this object.
    std::vector<std::shared_ptr<ov::AlignedBuffer>> m_sources;
};

}  // namespace ov::frontend::gguf
