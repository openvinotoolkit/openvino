// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "scaled_attn.hpp"

#include "shape_inference/shape_inference_cpu.hpp"
#include "shape_inference/shape_inference_ngraph.hpp"
#include "transformations/cpu_opset/common/op/sdpa.hpp"
#include "utils.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

class SDPAShapeInfer : public ShapeInferEmptyPads {
public:
    SDPAShapeInfer(const ScaledDotProductAttentionWithKVCache::Config& config) : m_config(config) {}

    IShapeInfer::Result infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
                              const std::unordered_map<size_t, MemoryPtr>& data_dependency) override {
        const auto& query = input_shapes.front().get();
        VectorDims past_kv = input_shapes.back().get();
        const auto& permute_axes = m_config.permute_axes;
        const size_t length_index = permute_axes.empty() ? query.size() - 2 : permute_axes[permute_axes.size() - 2];

        past_kv[length_index] += query[length_index];

        VectorDims output = query;
        if (!permute_axes.empty()) {
            // query needs permute to BHLS
            for (size_t i = 0; i < query.size(); i++) {
                output[i] = query[permute_axes[i]];
            }
        }
        return {{output, past_kv, past_kv}, ShapeInferStatus::success};
    }

    port_mask_t get_port_mask() const override {
        return EMPTY_PORT_MASK;
    }

private:
    ScaledDotProductAttentionWithKVCache::Config m_config;
};

ShapeInferPtr SDPAShapeInferFactory::makeShapeInfer() const {
    if (auto sdpa = std::dynamic_pointer_cast<const ScaledDotProductAttentionWithKVCache>(m_op)) {
        return std::make_shared<SDPAShapeInfer>(sdpa->get_config());
    } else {
        return std::make_shared<NgraphShapeInfer>(make_shape_inference(m_op), EMPTY_PORT_MASK);
    }
}

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
