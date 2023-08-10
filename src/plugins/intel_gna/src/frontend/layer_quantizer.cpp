// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_quantizer.hpp"

#include "backend/gna_types.hpp"
#include "common/gna_target.hpp"
#include "gna_graph_tools.hpp"
#include "weights_converter.hpp"

namespace ov {
namespace intel_gna {
using namespace limitations;
namespace frontend {

template <class T>
inline bool LayerQuantizer::ShouldAlwaysAllocate() {
    return false;
}

template <>
inline bool LayerQuantizer::ShouldAlwaysAllocate<gna_compound_bias_t>() {
    return true;
}

template <typename T>
InferenceEngine::Blob::Ptr LayerQuantizer::FP32ToPrecisionBlob(InferenceEngine::Blob::Ptr fp32_blob,
                                                               InferenceEngine::Precision precision,
                                                               QuantizationParams& dst_quant_params) {
    auto prec_blob = InferenceEngine::make_shared_blob<T>(
        {precision, fp32_blob->getTensorDesc().getDims(), fp32_blob->getTensorDesc().getLayout()});
    prec_blob->allocate();

    auto input_low = 0.0f;
    auto input_high = 0.0f;
    auto output_low = 0.0f;
    auto output_high = 0.0f;
    uint32_t levels = 1;

    if (dst_quant_params.IsStatsSet()) {
        input_low = dst_quant_params.GetMinValues(true).front();
        input_high = dst_quant_params.GetMaxValues(true).front();
        output_low = dst_quant_params.GetMinValues(false).front();
        output_high = dst_quant_params.GetMaxValues(false).front();
        levels = static_cast<uint32_t>(dst_quant_params.GetLevels());
    }

    auto f32_value_array = fp32_blob->buffer().as<float*>();

    for (auto& prec_value : *prec_blob) {
        auto f32_value = *f32_value_array++;

        if (dst_quant_params.IsStatsSet()) {
            f32_value = ApplyFQ(f32_value, input_low, input_high, output_low, output_high, levels);
        }

        f32_value = f32_value * dst_quant_params.GetScale();
        prec_value = SaturationCast<T>(f32_value);
    }

    return static_cast<InferenceEngine::Blob::Ptr>(prec_blob);
}

InferenceEngine::Blob::Ptr LayerQuantizer::FP32ToPrecisionBlob(InferenceEngine::Blob::Ptr fp32_blob,
                                                               InferenceEngine::Precision precision,
                                                               QuantizationParams& dst_quant_params) {
    InferenceEngine::Blob::Ptr result_ptr = nullptr;
    switch (precision) {
    case InferenceEngine::Precision::FP32:
        result_ptr = FP32ToPrecisionBlob<float>(fp32_blob, precision, dst_quant_params);
        break;
    case InferenceEngine::Precision::I32:
        result_ptr = FP32ToPrecisionBlob<int32_t>(fp32_blob, precision, dst_quant_params);
        break;
    case InferenceEngine::Precision::I16:
        result_ptr = FP32ToPrecisionBlob<int16_t>(fp32_blob, precision, dst_quant_params);
        break;
    case InferenceEngine::Precision::I8:
        result_ptr = FP32ToPrecisionBlob<int8_t>(fp32_blob, precision, dst_quant_params);
        break;
    default:
        THROW_GNA_EXCEPTION << "FP32 to " << precision << " not supported";
    }
    return result_ptr;
}

size_t LayerQuantizer::GetBiasSizeForLayer(InferenceEngine::WeightableLayer& wl) {
    if (wl._biases) {
        return wl._biases->size();
    } else if (LayerInfo(wl).isConvolution()) {
        // Calculating biases len using outdata dims: biases number should be equal to output channels number
        return InferenceEngine::GetDataDimSizeNHWC(wl.outData.front(), InferenceEngine::DataDimName::C);
    } else {
        // Calculating biases size using outData dimensions
        return wl.outData.front()->getDims().back();
    }
}

std::pair<size_t, size_t> LayerQuantizer::GetNumRowsColumns(InferenceEngine::WeightableLayer& wl) {
    size_t num_rows = 0;
    size_t num_columns = 0;

    if (LayerInfo(wl).isScaleShift()) {
        num_columns = 1;
        num_rows = wl._weights->size();
    } else if (LayerInfo(wl).isConvolution() || LayerInfo(wl).isConvolutionFilter()) {
        num_rows = GetBiasSizeForLayer(wl);

        if (num_rows == 0) {
            THROW_GNA_EXCEPTION << "Invalid nummber of rows";
        }

        num_columns = wl._weights->size() / num_rows;
    } else if (LayerInfo(wl).isAffineFilter() || LayerInfo(wl).isConcatAlignFilter()) {
        // For affine filter layer insdata size is not equal to the one stored in input layer
        num_rows = wl.outData.front()->getDims().back();
        num_columns = wl._weights->size() / num_rows;
    } else {
        num_rows = wl.outData.front()->getDims().back();
        num_columns = wl.insData.front().lock().get()->getDims().back();
    }

    return {num_rows, num_columns};
}

template <class WeightsType>
void LayerQuantizer::QuantizeWeightsPrep(InferenceEngine::WeightableLayer& wl, QuantizationData& common_data) {
    const auto weights_precision = InferenceEngine::Precision::fromType<WeightsType>();
    const auto int_weights = InferenceEngine::make_shared_blob<WeightsType>(
        InferenceEngine::TensorDesc(weights_precision,
                                    InferenceEngine::SizeVector({wl._weights->size()}),
                                    InferenceEngine::C));

    int_weights->allocate();

    if (int_weights->buffer() == nullptr) {
        IE_THROW(NotAllocated) << "[GNAPlugin] in function " << __PRETTY_FUNCTION__ << ": "
                               << "cannot copy weights for layer :" << wl.name << " of size" << int_weights->byteSize();
    }

    common_data.scale_factor = InferenceEngine::getInjectedData<QuantizedLayerParams>(wl)->_weights_quant.GetScale();
    const auto& blob_precision = wl._weights->getTensorDesc().getPrecision();
    const auto& quantized_weights =
        blob_precision != InferenceEngine::Precision::FP32 && blob_precision != InferenceEngine::Precision::FP16;
    const bool& compound_bias =
        IsBiasCompound(LayerInfo(wl), *InferenceEngine::getInjectedData<QuantizedLayerParams>(wl), gna_config);
    const auto& compound_bias_ptr =
        (compound_bias && wl._biases) ? wl._biases->buffer().as<gna_compound_bias_t*>() : nullptr;

    QuantizeWeights<WeightsType>(common_data,
                                 wl._weights->buffer().as<float*>(),
                                 int_weights->buffer(),
                                 compound_bias_ptr,
                                 quantized_weights);

    wl._weights = int_weights;

    // Correcting precision for outdata
    wl.precision = weights_precision;
}

void LayerQuantizer::QuantizeWeightsPrep(InferenceEngine::Precision precision,
                                         InferenceEngine::WeightableLayer& wl,
                                         QuantizationData& common_data) {
    switch (precision) {
    case InferenceEngine::Precision::I8:
        QuantizeWeightsPrep<int8_t>(wl, common_data);
        break;
    case InferenceEngine::Precision::I16:
        QuantizeWeightsPrep<int16_t>(wl, common_data);
        break;
    default:
        THROW_GNA_EXCEPTION << "Weights precision " << precision << " not supported!";
    }
}

template <class BiasesType>
void LayerQuantizer::QuantizeBiasesPrep(InferenceEngine::WeightableLayer& wl, QuantizationData& common_data) {
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

void LayerQuantizer::QuantizeBiasesPrep(InferenceEngine::Precision precision,
                                        InferenceEngine::WeightableLayer& wl,
                                        QuantizationData& common_data) {
    const auto compound_bias_precision = InferenceEngine::Precision::fromType<gna_compound_bias_t>();

    if (precision == InferenceEngine::Precision::I8) {
        QuantizeBiasesPrep<int8_t>(wl, common_data);
    } else if (precision == InferenceEngine::Precision::I16) {
        QuantizeBiasesPrep<int16_t>(wl, common_data);
    } else if (precision == InferenceEngine::Precision::I32) {
        QuantizeBiasesPrep<int32_t>(wl, common_data);
    } else if (precision == compound_bias_precision) {
        QuantizeBiasesPrep<gna_compound_bias_t>(wl, common_data);
    } else {
        THROW_GNA_EXCEPTION << "Biases precision " << precision << " not supported!";
    }
}

void LayerQuantizer::QuantizeWeightsBiases(InferenceEngine::WeightableLayer& wl) {
    float input_scale_factor = 1.f;

    if (InferenceEngine::CNNNetHasPrevLayer(&wl)) {
        auto quant_data_for_input_layer =
            InferenceEngine::getInjectedData<QuantizedLayerParams>(InferenceEngine::CNNNetPrevLayer(&wl));
        input_scale_factor = quant_data_for_input_layer->_dst_quant.GetScale();
        if (std::isnan(input_scale_factor) || std::isinf(input_scale_factor)) {
            IE_THROW() << "Unsupported input scale factor value " << input_scale_factor;
        }
    }

    size_t num_rows;
    size_t num_columns;

    std::tie(num_rows, num_columns) = GetNumRowsColumns(wl);

    auto quant_layer_params = InferenceEngine::getInjectedData<QuantizedLayerParams>(wl);

    QuantizationData common_data{num_rows, num_columns, kScaleFactorDefault, quant_layer_params->_weights_quant};

    auto bias_prec = GetBiasesPrecision(LayerInfo(wl), *quant_layer_params, gna_config);
    auto weight_prec = GetWeightsPrecision(LayerInfo(wl), *quant_layer_params, gna_config);

    QuantizeBiasesPrep(bias_prec, wl, common_data);
    QuantizeWeightsPrep(weight_prec, wl, common_data);

    // Correct precision for outdata
    for (auto&& outData : wl.outData) {
        outData->setPrecision(GetOutputPrecision());
    }
}

void LayerQuantizer::SetLayerOutputPrecision(InferenceEngine::CNNLayer& cnn_layer) {
    // Set scale factor for input layers
    if (cnn_layer.insData.empty() || LayerInfo(cnn_layer).isCrop() || LayerInfo(cnn_layer).isConcat() ||
        LayerInfo(cnn_layer).isSplit() || LayerInfo(cnn_layer).isActivation() || LayerInfo(cnn_layer).isCopy() ||
        LayerInfo(cnn_layer).isNonFunctional() || LayerInfo(cnn_layer).isPermute() || LayerInfo(cnn_layer).isConst() ||
        LayerInfo(cnn_layer).isMaxPooling()) {
        // Precision of activation and pooling layers is always equal input precision
        for (auto&& out_data : cnn_layer.outData) {
            out_data->setPrecision(GetInputPrecision());
        }
    } else {
        for (auto&& out_data : cnn_layer.outData) {
            out_data->setPrecision(GetOutputPrecision());
        }
    }
}

void LayerQuantizer::CreateConstBlob(InferenceEngine::CNNLayer& cnn_layer) {
    auto initial_precision = cnn_layer.blobs["custom"]->getTensorDesc().getPrecision();
    IE_ASSERT(initial_precision != InferenceEngine::Precision::I32);

    if (initial_precision == InferenceEngine::Precision::FP16) {
        cnn_layer.blobs["custom"] = make_fp32_blob(cnn_layer.blobs["custom"]);
    }

    auto quant_params = InferenceEngine::getInjectedData<QuantizedLayerParams>(cnn_layer);
    auto new_const_blob = InferenceEngine::Blob::CreateFromData(cnn_layer.outData.front());
    auto const_blob = cnn_layer.blobs["custom"];

    if (const_blob->getTensorDesc().getPrecision() == InferenceEngine::Precision::FP32) {
        cnn_layer.blobs["custom"] =
            FP32ToPrecisionBlob(const_blob, cnn_layer.outData.front()->getPrecision(), quant_params->_dst_quant);
    }
}

void LayerQuantizer::quantize(InferenceEngine::CNNLayer& layer) {
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

LayerQuantizer::LayerQuantizer(const Config& gna_config) : gna_config(gna_config) {}

InferenceEngine::Precision LayerQuantizer::GetOutputPrecision() {
    return InferenceEngine::Precision::I32;
}

InferenceEngine::Precision GetBiasesPrecision(const LayerInfo& layer_info,
                                              const QuantizedLayerParams& quant_layer_params,
                                              const Config& gna_config) {
    if (layer_info.isConvolution() || layer_info.isConvolutionFilter() || layer_info.isScaleShift()) {
        return InferenceEngine::Precision::I32;
    }

    if (quant_layer_params._weights_quant.IsStatsSet()) {
        // For networks with FakeQuantize layers
        const auto fq_levels = quant_layer_params._weights_quant.GetLevels();

        if (fq_levels <= std::numeric_limits<uint8_t>::max()) {
            return InferenceEngine::Precision::fromType<gna_compound_bias_t>();
        } else {
            return InferenceEngine::Precision::I32;
        }
    } else {
        if (gna_config.gnaFlags.input_low_precision) {
            return InferenceEngine::Precision::I8;
        } else {
            return (GetWeightsPrecision(layer_info, quant_layer_params, gna_config) == InferenceEngine::Precision::I8)
                       ? InferenceEngine::Precision::fromType<gna_compound_bias_t>()
                       : InferenceEngine::Precision{InferenceEngine::Precision::I32};
        }
    }
}

InferenceEngine::Precision GetInputPrecision() {
    return InferenceEngine::Precision::I16;
}

InferenceEngine::Precision GetWeightsPrecision(const LayerInfo& layer_info,
                                               const QuantizedLayerParams& quant_layer_params,
                                               const Config& gna_config) {
    if (((layer_info.isConvolution() || layer_info.isConvolutionFilter()) &&
         Limitations::get_instance()->use_only_16bit_convolution_weights()) ||
        layer_info.isScaleShift()) {
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
        return gna_config.gnaPrecision;
    }
}

bool IsBiasCompound(const LayerInfo& layer_info,
                    const QuantizedLayerParams& quant_layer_params,
                    const Config& gna_config) {
    auto biases_precision = GetBiasesPrecision(layer_info, quant_layer_params, gna_config);
    auto compound_bias_precision = InferenceEngine::Precision::fromType<gna_compound_bias_t>();
    return (biases_precision == compound_bias_precision);
}

}  // namespace frontend
}  // namespace intel_gna
}  // namespace ov
