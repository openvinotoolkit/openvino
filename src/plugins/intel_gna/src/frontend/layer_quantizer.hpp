// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ie_precision.hpp"
#include "layers/gna_layer_info.hpp"
#include "quantized_layer_params.hpp"
#include "backend/gna_types.h"
#include "quantization.hpp"
#include "weights_converter.hpp"


namespace GNAPluginNS {
namespace frontend {

// TODO: POT AAQ support can be added by modifying these functions
static InferenceEngine::Precision GetInputPrecision() {
    return InferenceEngine::Precision::I16;
}

static InferenceEngine::Precision GetOutputPrecision() {
    return InferenceEngine::Precision::I32;
}

// TODO: These should take into account the HW platform limitations
static InferenceEngine::Precision GetWeightsPrecision(const LayerInfo& layer_info,
                                                      const QuantizedLayerParams& quant_layer_params) {
    if (layer_info.isConvolution() || layer_info.isConvolutionFilter() || layer_info.isScaleShift()) {
        return InferenceEngine::Precision::I16;
    }

    if (quant_layer_params._weights_quant.IsStatsSet()) {
        // For networks with FakeQuantize layers
        if (quant_layer_params._weights_quant.GetLevels() <= std::numeric_limits<uint8_t>::max()) {
            return InferenceEngine::Precision::I8;
        } else {
            return InferenceEngine::Precision::I16;
        }
    } else {
        return quant_layer_params._weights_int8_precision ? InferenceEngine::Precision::I8
                                                          : InferenceEngine::Precision::I16;
    }
}

// TODO: In future these should take into account the HW platform limitations
static InferenceEngine::Precision GetBiasesPrecision(const LayerInfo& layer_info,
                                                     const QuantizedLayerParams& quant_layer_params) {
    if (layer_info.isConvolution() || layer_info.isConvolutionFilter() || layer_info.isScaleShift()) {
        return InferenceEngine::Precision::I32;
    }

    if (quant_layer_params._weights_quant.IsStatsSet()) {
        // For networks with FakeQuantize layers
        const auto fq_levels = quant_layer_params._weights_quant.GetLevels();

        if (fq_levels <= std::numeric_limits<uint8_t>::max()) {
            return InferenceEngine::Precision::fromType<gna_compound_bias_t>();
            // TODO: enable when all available precisions should be supported; it will require tests modifications
            //} else if (fqLevels <= std::numeric_limits<uint16_t>::max()) {
            //    return InferenceEngine::Precision::I16;
            //} else if (GetInputPrecision() != InferenceEngine::Precision::I16 ||
            //    GetWeightsPrecision(layer_info, quant_data) != InferenceEngine::Precision::I8) {
        } else {
            return InferenceEngine::Precision::I32;
        }
    } else {
        if (quant_layer_params._inputs_int8_precision) {
            return InferenceEngine::Precision::I8;
        } else {
            return (GetWeightsPrecision(layer_info, quant_layer_params) == InferenceEngine::Precision::I8)
                       ? InferenceEngine::Precision::fromType<gna_compound_bias_t>()
                       : InferenceEngine::Precision{InferenceEngine::Precision::I32};
        }
    }
}

static bool IsBiasCompound(const LayerInfo& layer_info, const QuantizedLayerParams* quant_layer_params) {
    auto biases_precision = GetBiasesPrecision(layer_info, *quant_layer_params);
    auto compound_bias_precision = InferenceEngine::Precision::fromType<gna_compound_bias_t>();
    return (biases_precision == compound_bias_precision ? true : false);
}

struct FakeQuantizeParams {
    bool params_set = false;
    uint32_t levels_num = 1;
    float input_min_value = 1.0f;
    float input_max_value = 1.0f;
    float output_min_value = 1.0f;
    float output_max_value = 1.0f;
};

/**
 * @brief should allocated blob for specific data type, in case of src blob is nullptr
 * @tparam T
 * @return
 */
template <class T>
inline bool ShouldAlwaysAllocate() {
    return false;
}

template <>
inline bool ShouldAlwaysAllocate<gna_compound_bias_t>() {
    return true;
}

template <typename T>
inline InferenceEngine::Blob::Ptr FP32ToPrecisionBlob(InferenceEngine::Blob::Ptr fp32_blob,
                                                      InferenceEngine::Precision precision,
                                                      float scale_factor,
                                                      const FakeQuantizeParams& fq_params) {
    auto prec_blob = InferenceEngine::make_shared_blob<T>(
        {precision, fp32_blob->getTensorDesc().getDims(), fp32_blob->getTensorDesc().getLayout()});
    prec_blob->allocate();

    auto input_low = 0.0f;
    auto input_high = 0.0f;
    auto output_low = 0.0f;
    auto output_high = 0.0f;
    auto levels = 1;
    if (fq_params.params_set) {
        input_low = fq_params.input_min_value;
        input_high = fq_params.input_max_value;
        output_low = fq_params.output_min_value;
        output_high = fq_params.output_max_value;
        levels = fq_params.levels_num;
    }

    int i = 0;

    auto f32_value_array =
        fp32_blob->buffer()
            .template as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();

    for (auto& prec_value : *prec_blob) {
        auto f32_value = f32_value_array[i++];

        if (fq_params.params_set) {
            auto x = f32_value;
            if (x <= std::min(input_low, input_high)) {
                f32_value = output_low;
            } else if (x > std::max(input_low, input_high)) {
                f32_value = output_high;
            } else {
                f32_value = nearbyint((x - input_low) / (input_high - input_low) * (levels - 1)) / (levels - 1) *
                                (output_high - output_low) +
                            output_low;
            }
        }

        f32_value = f32_value * scale_factor;
        if (f32_value > static_cast<float>(std::numeric_limits<T>::max())) {
            prec_value = std::numeric_limits<T>::max();
        } else if (f32_value < std::numeric_limits<T>::min()) {
            prec_value = std::numeric_limits<T>::min();
        } else {
            prec_value = static_cast<T>(f32_value);
        }
    }

    return static_cast<InferenceEngine::Blob::Ptr>(prec_blob);
}

inline InferenceEngine::Blob::Ptr FP32ToPrecisionBlob(InferenceEngine::Blob::Ptr fp32_blob,
                                                      InferenceEngine::Precision precision,
                                                      float scale_factor,
                                                      const FakeQuantizeParams& fq_params) {
    InferenceEngine::Blob::Ptr result_ptr = nullptr;
    switch (precision) {
    case InferenceEngine::Precision::FP32:
        result_ptr = FP32ToPrecisionBlob<float>(fp32_blob, precision, scale_factor, fq_params);
        break;
    case InferenceEngine::Precision::I32:
        result_ptr = FP32ToPrecisionBlob<int32_t>(fp32_blob, precision, scale_factor, fq_params);
        break;
    case InferenceEngine::Precision::I16:
        result_ptr = FP32ToPrecisionBlob<int16_t>(fp32_blob, precision, scale_factor, fq_params);
        break;
    case InferenceEngine::Precision::I8:
        result_ptr = FP32ToPrecisionBlob<int8_t>(fp32_blob, precision, scale_factor, fq_params);
        break;
    default:
        THROW_GNA_EXCEPTION << "FP32 to " << precision << " not supported";
    }
    return result_ptr;
}

static size_t GetBiasSizeForLayer(InferenceEngine::WeightableLayer& wl) {
    if (wl._biases) {
        return wl._biases->size();
    } else if (LayerInfo(wl).isConvolution()) {
        // calculating biases len using outdata dims: biases number should be equal to output channels number
        size_t size = InferenceEngine::GetDataDimSize(wl.outData.front(), InferenceEngine::DataDimName::C);
        return size;
    } else {
        // calculating biases size using outData dimensions
        const auto& dims = wl.outData.front()->getDims();
        const auto& o_idx = wl.outData[0]->getDims().size() - 1;
        return dims[o_idx];
    }
}

static std::pair<size_t, size_t> GetNumRowsColumns(InferenceEngine::WeightableLayer& wl) {
    size_t num_rows;
    size_t num_columns;
    const auto& o_idx = wl.outData[0]->getDims().size() - 1;
    const auto& i_idx = wl.insData[0].lock().get()->getDims().size() - 1;

    if (LayerInfo(wl).isScaleShift()) {
        num_columns = 1;
        num_rows = wl._weights->size();
    } else if (LayerInfo(wl).isConvolution() || LayerInfo(wl).isConvolutionFilter()) {
        if (wl.outData[0]->getDims().size() < 2) {
            IE_THROW() << "Unsupported output dims size for " << wl.name << ". Should be > 1, but is "
                       << wl.outData[0]->getDims().size();
        }
        if (wl.insData[0].lock().get()->getDims().size() < 2) {
            IE_THROW() << "Unsupported input dims size for " << wl.name << ". Should be > 1, but is "
                       << wl.insData[0].lock().get()->getDims().size();
        }

        num_rows = GetBiasSizeForLayer(wl);

        if (num_rows == 0) {
            THROW_GNA_EXCEPTION << "Invalid nummber of rows";
        }

        num_columns = wl._weights->size() / num_rows;
    } else if (LayerInfo(wl).isAffineFilter() || LayerInfo(wl).isConcatAlignFilter()) {
        // For affine filter layer insdata size is not equal to the one stored in input layer
        num_rows = wl.outData[0]->getDims()[o_idx];
        num_columns = wl._weights->size() / num_rows;
    } else {
        num_rows = wl.outData[0]->getDims()[o_idx];
        num_columns = wl.insData[0].lock().get()->getDims()[i_idx];
    }

    return {num_rows, num_columns};
}

template <class WeightsType>
void QuantizeWeightsPrep(InferenceEngine::WeightableLayer& wl, QuantizationData& common_data) {
    const auto weights_precision = InferenceEngine::Precision::fromType<WeightsType>();
    const auto int_weights = InferenceEngine::make_shared_blob<WeightsType>(
        InferenceEngine::TensorDesc(weights_precision,
                                    InferenceEngine::SizeVector({wl._weights->size()}),
                                    InferenceEngine::C));

    int_weights->allocate();

    if (int_weights->buffer() == nullptr) {
        IE_THROW(NotAllocated) << "[GNAPlugin] in function " << __PRETTY_FUNCTION__ << ": "
                               << "cannot copy weights for layer :" << wl.name << " of size"
                               << int_weights->byteSize();
    }

    common_data.scale_factor = InferenceEngine::getInjectedData<QuantizedLayerParams>(wl)->_weights_quant.GetScale();
    const auto& blob_precision = wl._weights->getTensorDesc().getPrecision();
    const auto& quantized_weights =
        blob_precision != InferenceEngine::Precision::FP32 && blob_precision != InferenceEngine::Precision::FP16;
    const bool& compound_bias =
        IsBiasCompound(LayerInfo(wl), InferenceEngine::getInjectedData<QuantizedLayerParams>(wl));
    const auto& compound_bias_ptr =
        (compound_bias && wl._biases) ? wl._biases->buffer().as<gna_compound_bias_t*>() : nullptr;

    QuantizeWeights<WeightsType>(common_data,
                                 wl._weights->buffer().as<float*>(),
                                 int_weights->buffer(),
                                 compound_bias_ptr,
                                 quantized_weights);

    wl._weights = int_weights;

    /**
     * correcting precision for outdata
     */
    wl.precision = weights_precision;
}

template <class BiasesType>
void QuantizeBiasesPrep(InferenceEngine::WeightableLayer& wl, QuantizationData& common_data) {
    const auto bias_maker = [&]() {
        InferenceEngine::Blob::Ptr zero;
        if (!wl._biases && !ShouldAlwaysAllocate<BiasesType>()) {
            return zero;
        }

        InferenceEngine::Blob::Ptr bias = InferenceEngine::make_shared_blob<BiasesType>(
            InferenceEngine::TensorDesc(InferenceEngine::Precision::fromType<BiasesType>(),
                                        InferenceEngine::SizeVector({GetBiasSizeForLayer(wl)}),
                                        InferenceEngine::C));

        bias->allocate();

        if (bias->buffer() == nullptr) {
            IE_THROW(NotAllocated) << "[GNAPlugin] in function " << __PRETTY_FUNCTION__ << ": "
                                   << "cannot copy bias for layer :" << wl.name << "of size" << bias->byteSize();
        }

        memset(bias->buffer(), 0, bias->byteSize());

        return bias;
    };

    const auto int_biases = bias_maker();
    common_data.scale_factor = InferenceEngine::getInjectedData<QuantizedLayerParams>(wl)->_dst_quant.GetScale();

    QuantizeBiases<BiasesType>(common_data,
                               wl._biases ? wl._biases->buffer().as<float*>() : nullptr,
                               int_biases ? int_biases->buffer() : static_cast<BiasesType*>(nullptr));

    wl._biases = int_biases;
}

static const std::map<InferenceEngine::Precision, void (*)(InferenceEngine::WeightableLayer&, QuantizationData&)> GetWeightsPreicisionsMap() {
    return {{InferenceEngine::Precision::I8, QuantizeWeightsPrep<int8_t>},
            {InferenceEngine::Precision::I16, QuantizeWeightsPrep<int16_t>}};
}

static const std::map<InferenceEngine::Precision, void (*)(InferenceEngine::WeightableLayer&, QuantizationData&)> GetBiasPreicisionsMap() {
    return {{InferenceEngine::Precision::I8, QuantizeBiasesPrep<int8_t>},
            {InferenceEngine::Precision::I16, QuantizeBiasesPrep<int16_t>},
            {InferenceEngine::Precision::I32, QuantizeBiasesPrep<int32_t>},
            {InferenceEngine::Precision::fromType<gna_compound_bias_t>(), QuantizeBiasesPrep<gna_compound_bias_t>}};
}

static void QuantizeWeightsBiases(InferenceEngine::WeightableLayer* wl) {
    float input_scale_factor = 1.f;

    if (InferenceEngine::CNNNetHasPrevLayer(wl)) {
        auto quant_data_for_input_layer =
            InferenceEngine::getInjectedData<QuantizedLayerParams>(InferenceEngine::CNNNetPrevLayer(wl));
        input_scale_factor = quant_data_for_input_layer->_dst_quant.GetScale();
        if (std::isnan(input_scale_factor) ||
            std::isinf(input_scale_factor)) {
            IE_THROW() << "Unsupported input scale factor value " << input_scale_factor;
        }
    }

    size_t num_rows;
    size_t num_columns;

    std::tie(num_rows, num_columns) = GetNumRowsColumns(*wl);

    auto quant_layer_params = InferenceEngine::getInjectedData<QuantizedLayerParams>(*wl);

    QuantizationData common_data{
        num_rows,
        num_columns,
        1.0f,
        quant_layer_params->_weights_quant
    };

    GetBiasPreicisionsMap().at(GetBiasesPrecision(LayerInfo(wl), *quant_layer_params))(*wl, common_data);
    GetWeightsPreicisionsMap().at(GetWeightsPrecision(LayerInfo(wl), *quant_layer_params))(*wl, common_data);

    // Correct precision for outdata
    for (auto &&outData : wl->outData) {
        outData->setPrecision(GetOutputPrecision());
    }
}

/**
 * Helper class to use partial specialisation of Layer type
 * @tparam Layer
 */
template<class Layer>
class DataQuantizer {
 public:
    bool operator()(Layer cnn_layer) const {
        return false;
    }
};

template<>
class DataQuantizer<InferenceEngine::CNNLayer*> {
public:
    bool operator()(InferenceEngine::CNNLayer* cnn_layer) const {
        for (auto&& out_data : cnn_layer->outData) {
            out_data->setPrecision(GetOutputPrecision());
        }

        // Set scale factor for input layers
        if (cnn_layer->insData.empty() || LayerInfo(*cnn_layer).isCrop() || LayerInfo(*cnn_layer).isConcat() ||
            LayerInfo(*cnn_layer).isSplit()) {
            for (auto&& out_data : cnn_layer->outData) {
                out_data->setPrecision(GetInputPrecision());
            }
        } else {
            if (LayerInfo(*cnn_layer).isActivation() || LayerInfo(*cnn_layer).isCopy() ||
                LayerInfo(*cnn_layer).isNonFunctional() || LayerInfo(*cnn_layer).isPermute() ||
                LayerInfo(*cnn_layer).isConst()) {
                // Precision of activation layers is always equal input precision
                for (auto&& out_data : cnn_layer->outData) {
                    out_data->setPrecision(GetInputPrecision());
                }
            }
            // For pooling layer output precision is the same as input precision
            if (LayerInfo(*cnn_layer).isMaxPooling()) {
                const auto inputPrecision = cnn_layer->insData.front().lock()->getPrecision();
                for (auto&& out_data : cnn_layer->outData) {
                    out_data->setPrecision(inputPrecision);
                }
            }
        }

        cnn_layer->precision = GetInputPrecision();

        if (LayerInfo(*cnn_layer).isConst()) {
            auto initial_precision = cnn_layer->blobs["custom"]->getTensorDesc().getPrecision();
            // TODO: I32 must be handled separately when it's supported
            IE_ASSERT(initial_precision != InferenceEngine::Precision::I32);

            if (initial_precision == InferenceEngine::Precision::FP16) {
                cnn_layer->blobs["custom"] = make_fp32_blob(cnn_layer->blobs["custom"]);
            }
            auto quant_params = InferenceEngine::getInjectedData<QuantizedLayerParams>(*cnn_layer);
            auto new_const_blob = InferenceEngine::Blob::CreateFromData(cnn_layer->outData[0]);
            auto const_blob = cnn_layer->blobs["custom"];
            if (const_blob->getTensorDesc().getPrecision() == InferenceEngine::Precision::FP32) {
                auto fq_params = FakeQuantizeParams{};
                if (quant_params->_dst_quant.IsStatsSet()) {
                    fq_params.params_set = true;
                    fq_params.levels_num = quant_params->_dst_quant.GetLevels();
                    fq_params.input_min_value = quant_params->_dst_quant.GetMinValues(true).front();
                    fq_params.input_max_value = quant_params->_dst_quant.GetMaxValues(true).front();
                    fq_params.output_min_value = quant_params->_dst_quant.GetMinValues(false).front();
                    fq_params.output_max_value = quant_params->_dst_quant.GetMaxValues(false).front();
                }

                cnn_layer->blobs["custom"] = FP32ToPrecisionBlob(const_blob,
                                                                   cnn_layer->outData[0]->getPrecision(),
                                                                   quant_params->_dst_quant.GetScale(),
                                                                   fq_params);
            } else if (LayerInfo(*cnn_layer).isReshape()) {
                for (auto&& out_data : cnn_layer->outData) {
                    out_data->setPrecision(cnn_layer->insData.front().lock()->getPrecision());
                }
            }
        }

        return true;
    }
};

template<>
class DataQuantizer<InferenceEngine::WeightableLayer*> {
 public:
    bool operator()(InferenceEngine::WeightableLayer* wl) const {
        QuantizeWeightsBiases(wl);
        return true;
    }
};

}  // namespace frontend

struct LayersQuantizer {
    explicit LayersQuantizer(float scaleFactor) {}
    template<class T>
    bool operator()(T input) const {
        return frontend::DataQuantizer<T>()(input);
    }
};

enum class QuantizedDataType {
    input,
    output,
    weights,
    bias
};

/**
 * @brief Returns a scale factor for specific layer data
 * @param layer Layer to be quantized
 * @param data_type Type of data to be quantized
 * @return scale factor
 */
inline float GetScaleFactor(InferenceEngine::CNNLayerPtr layer, QuantizedDataType data_type) {
    IE_ASSERT(layer != nullptr);
    auto quantized = InferenceEngine::getInjectedData<QuantizedLayerParams>(layer);
    float scale_factor;
    if (!quantized) {
        scale_factor = 1.0f;
    } else {
        switch (data_type) {
            case QuantizedDataType::input:
                scale_factor = quantized->_src_quant.GetScale();
                break;
            case QuantizedDataType::output:
                scale_factor = quantized->_dst_quant.GetScale();
                break;
            case QuantizedDataType::weights:
                scale_factor = quantized->_weights_quant.GetScale();
                break;
            case QuantizedDataType::bias:
                scale_factor = quantized->_bias_quant.GetScale();
                break;
            default:
                THROW_GNA_LAYER_EXCEPTION(layer) << "Unsupported data type for quantization: " << static_cast<int>(data_type);
        }
    }

    if (scale_factor <= 0.0 || std::isinf(scale_factor)) {
        THROW_GNA_LAYER_EXCEPTION(layer) << "Invalid scale factor: " << scale_factor;
    }

    return scale_factor;
}

}  // namespace GNAPluginNS
