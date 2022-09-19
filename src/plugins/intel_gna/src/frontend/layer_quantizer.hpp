// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "descriptions/gna_desc.hpp"
#include "layers/gna_layer_info.hpp"
#include "quantized_layer_params.hpp"
#include "quantization.hpp"

using namespace GNAPluginNS;

namespace ov {
namespace intel_gna {
namespace frontend {

/**
 * @brief Returns layer's target input precision
 * @return layer's target input precision
 */
InferenceEngine::Precision GetInputPrecision();

/**
 * @brief Returns layer's target weights precision
 * @param layer_info Layer's information
 * @param quant_layer_params Layer's quantization parameters
 * @return layer's target weights precision
 */
InferenceEngine::Precision GetWeightsPrecision(const LayerInfo& layer_info,
                                               const QuantizedLayerParams& quant_layer_params,
                                               const Config& gna_config);

class LayerQuantizer {
    const Config& gna_config;

    void QuantizeWeightsBiases(InferenceEngine::WeightableLayer& wl);
    void SetLayerOutputPrecision(InferenceEngine::CNNLayer& cnn_layer);
    void CreateConstBlob(InferenceEngine::CNNLayer& cnn_layer);
    std::pair<size_t, size_t> GetNumRowsColumns(InferenceEngine::WeightableLayer& wl);
    InferenceEngine::Blob::Ptr FP32ToPrecisionBlob(InferenceEngine::Blob::Ptr fp32_blob,
                                                   InferenceEngine::Precision precision,
                                                   QuantizationParams& dst_quant_params);

    template <typename T>
    InferenceEngine::Blob::Ptr FP32ToPrecisionBlob(InferenceEngine::Blob::Ptr fp32_blob,
                                                   InferenceEngine::Precision precision,
                                                   QuantizationParams& dst_quant_params);

    template <class WeightsType>
    void QuantizeWeightsPrep(InferenceEngine::WeightableLayer& wl, QuantizationData& common_data);

    void QuantizeWeightsPrep(InferenceEngine::Precision precision,
                            InferenceEngine::WeightableLayer& wl,
                            QuantizationData& common_data);

    template <class BiasesType>
    void QuantizeBiasesPrep(InferenceEngine::WeightableLayer& wl, QuantizationData& common_data);

    void QuantizeBiasesPrep(InferenceEngine::Precision precision,
                            InferenceEngine::WeightableLayer& wl,
                            QuantizationData& common_data);

    template <class T>
    inline bool ShouldAlwaysAllocate();

    size_t GetBiasSizeForLayer(InferenceEngine::WeightableLayer& wl);
    bool IsBiasCompound(const LayerInfo& layer_info, const QuantizedLayerParams* quant_layer_params);
    InferenceEngine::Precision GetOutputPrecision();
    InferenceEngine::Precision GetBiasesPrecision(const LayerInfo& layer_info,
                                                  const QuantizedLayerParams& quant_layer_params);

public:
    LayerQuantizer(const Config& gna_config) : gna_config(gna_config) {}

    void quantize(InferenceEngine::CNNLayer& layer) {
        auto layer_info = LayerInfo(layer);

        if (layer_info.isWeightable()) {
            QuantizeWeightsBiases(dynamic_cast<InferenceEngine::WeightableLayer&>(layer));
        } else {
            layer.precision = GetInputPrecision();

            SetLayerOutputPrecision(layer);

            if (layer_info.isConst()) {
                CreateConstBlob(layer);
            }
        }
    }
};

template <>
inline bool LayerQuantizer::ShouldAlwaysAllocate<gna_compound_bias_t>();

}  // namespace frontend
}  // namespace intel_gna
}  // namespace ov
