// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>

#include "model_builder_types.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov {
namespace test {
namespace npuw {

struct SwiGLU {
    size_t hidden_size;
    size_t intermediate_size;
    ov::element::Type precision;
    WeightFn weight_fn;
    const LoRAInjector* lora = nullptr;

    SwiGLU(size_t hs, size_t is, ov::element::Type prec, WeightFn wf, const LoRAInjector* l = nullptr)
        : hidden_size(hs),
          intermediate_size(is),
          precision(prec),
          weight_fn(std::move(wf)),
          lora(l) {}

    ov::Output<ov::Node> operator()(const ov::Output<ov::Node>& input, const std::string& name) const;
};

struct GELU {
    size_t hidden_size;
    size_t intermediate_size;
    ov::element::Type precision;
    WeightFn weight_fn;
    WeightFn bias_fn;
    const LoRAInjector* lora = nullptr;

    GELU(size_t hs, size_t is, ov::element::Type prec, WeightFn wf, WeightFn bf = {}, const LoRAInjector* l = nullptr)
        : hidden_size(hs),
          intermediate_size(is),
          precision(prec),
          weight_fn(std::move(wf)),
          bias_fn(std::move(bf)),
          lora(l) {}

    ov::Output<ov::Node> operator()(const ov::Output<ov::Node>& input, const std::string& name) const;
};

}  // namespace npuw
}  // namespace test
}  // namespace ov
