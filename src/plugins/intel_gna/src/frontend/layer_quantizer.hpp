// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "descriptions/gna_desc.hpp"
#include "layers/gna_layer_info.hpp"
#include "quantized_layer_params.hpp"
#include "quantization.hpp"

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
 * @param gna_config GNA Plugin configuration
 * @return layer's target weights precision
 */
InferenceEngine::Precision GetWeightsPrecision(const LayerInfo& layer_info,
                                               const QuantizedLayerParams& quant_layer_params,
                                               const Config& gna_config);

/**
 * @brief Returns layer's target biases precision
 * @param layer_info Layer's information
 * @param quant_layer_params Layer's quantization parameters
 * @param gna_config GNA Plugin configuration
 * @return layer's target biases precision
 */
InferenceEngine::Precision GetBiasesPrecision(const LayerInfo& layer_info,
                                              const QuantizedLayerParams& quant_layer_params,
                                              const Config& gna_config);

/**
 * @brief Checks whether layer's target biases are compound
 * @param layer_info Layer's information
 * @param quant_layer_params Layer's quantization parameters
 * @param gna_config GNA Plugin configuration
 * @return true if layer's target biases are compound
 */
bool IsBiasCompound(const LayerInfo& layer_info,
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
    InferenceEngine::Precision GetOutputPrecision();

public:
    LayerQuantizer(const Config& gna_config);

    void quantize(InferenceEngine::CNNLayer& layer);
};

template <>
inline bool LayerQuantizer::ShouldAlwaysAllocate<gna_compound_bias_t>();

}  // namespace frontend
}  // namespace intel_gna
}  // namespace ov
