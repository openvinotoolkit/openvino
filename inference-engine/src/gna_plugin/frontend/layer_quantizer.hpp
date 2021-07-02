// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <utility>
#include <cmath>

#include "backend/gna_types.h"
#include "gna_plugin_log.hpp"
#include "quantized_layer_params.hpp"
#include "quantization.h"
#include "gna_graph_tools.hpp"
#include "blob_factory.hpp"
#include "precision_ex.hpp"
#include "layers/gna_layer_info.hpp"
#include "weights_converter.hpp"
#include <legacy/layer_transform.hpp>

namespace GNAPluginNS {
namespace frontend {

/**
 * @brief description of quantisation precision
 * @tparam Ip - input precision
 * @tparam Op - output precision
 * @tparam Wp - weights precision
 * @tparam Bp - biases precision
 * @tparam Np - network precision - can be auto generated in future
 */
template <class Ip, class Op, class Wp, class Bp, class Np>
struct QuantDescTmpl {
    using WeightsPrecision = Wp;
    using BiasesPrecision = Bp;

    InferenceEngine::TPrecision<Ip> _Ip;
    InferenceEngine::TPrecision<Op> _Op;
    InferenceEngine::TPrecision<Wp> _Wp;
    InferenceEngine::TPrecision<Bp> _Bp;
    InferenceEngine::TPrecision<Np> _Np;

    QuantDescTmpl() = default;
    QuantDescTmpl(InferenceEngine::TPrecision<Ip> _Ip,
              InferenceEngine::TPrecision<Op> _Op,
              InferenceEngine::TPrecision<Wp> _Wp,
              InferenceEngine::TPrecision<Bp> _Bp,
              InferenceEngine::TPrecision<Np> _Np) : _Op(_Op), _Ip(_Ip), _Wp(_Wp), _Bp(_Bp), _Np(_Np) {
    }

    InferenceEngine::Precision getInputPrecision() const {
        return _Ip;
    }
    InferenceEngine::Precision getWeightsPrecision() const {
        return _Wp;
    }
    InferenceEngine::Precision getBiasesPrecision() const {
        return _Bp;
    }
    InferenceEngine::Precision getNetPrecision() const {
        return _Np;
    }
    InferenceEngine::Precision getOutputPrecision() const {
        return _Op;
    }
};

#define P_TYPE(X)\
typename InferenceEngine::PrecisionTrait<InferenceEngine::Precision::X>::value_type

#define PRECISION_TYPE(A, B, C, D, E)\
    P_TYPE(A), P_TYPE(B), P_TYPE(C), P_TYPE(D), P_TYPE(E)


struct QuantI16 : public QuantDescTmpl<PRECISION_TYPE(I16, I32, I16, I32, MIXED)> {
    QuantI16() {
        _Np = InferenceEngine::Precision::MIXED;
    }
};
struct QuantI8  : public QuantDescTmpl<P_TYPE(I16), P_TYPE(I32), P_TYPE(I8), gna_compound_bias_t, P_TYPE(MIXED)> {
    QuantI8() {
        _Np = InferenceEngine::Precision::MIXED;
    }
};
// Low precision path quantizer (I8 inputs, weights, biases)
struct QuantI8_I8 : public QuantDescTmpl<PRECISION_TYPE(I8, I32, I8, I8, MIXED)> {
    QuantI8_I8() {
        _Np = InferenceEngine::Precision::MIXED;
    }
};

// for support proper trait instantiation for quantization function callback
struct FakeQuantI16 : public QuantI16 {};
struct FakeQuantI8 : public QuantI8 {};

template <class A, class B>
struct QuantPair {
    using MandatoryType = A;
    using OptionalType = B;
    static A mandatory () { return A();}
    static B optional () { return B();}
};

struct FakeQuantizeParams {
    bool paramsSet = false;
    uint32_t levelsNum = 1;
    float inputMinValue = 1.0f;
    float inputMaxValue = 1.0f;
    float outputMinValue = 1.0f;
    float outputMaxValue = 1.0f;
};

/**
 * @brief should allocated blob for specific data type, in case of src blob is nullptr
 * @tparam T
 * @return
 */
template <class T>
inline bool shouldAlwaysAllocate() {
    return false;
}

template <>
inline bool shouldAlwaysAllocate<gna_compound_bias_t>() {
    return true;
}


#undef P_TYPE
#undef PRECISION_TYPE

/**
 * @brief  designate actual data quantisation functions trait
 */
template <class T>
class Quant {
public:
    template<class ...Args>
    void operator()(Args && ... args) const { }
};

template<>
class Quant<QuantI16> {
 public:
    template<class ...Args>
    void operator()(Args && ... args) const {
        QuantizationCallback<int16_t, int32_t> {
            std::forward<Args>(args)...
        }.runQuantize();
    }
};

template<>
class Quant<QuantI8> {
 public:
    template<class ...Args>
    void operator()(Args && ... args) const {
        QuantizationCallback<int8_t, gna_compound_bias_t> {
            std::forward<Args>(args)...
        }.runQuantize();
    }
};

template<>
class Quant<QuantI8_I8> {
public:
    template<class ...Args>
    void operator()(Args && ... args) const {
        QuantizationCallback<int8_t, int8_t> {
            std::forward<Args>(args)...
        }.runQuantize();
    }
};

template<>
class Quant<FakeQuantI16> {
 public:
    template<class ...Args>
    void operator()(Args && ... args) const {
        QuantizationCallback<int16_t, int32_t> {
            std::forward<Args>(args)...
        }.runFakeQuantize();
    }
};

template<>
class Quant<FakeQuantI8> {
 public:
    template<class ...Args>
    void operator()(Args && ... args) const {
        QuantizationCallback<int8_t, gna_compound_bias_t>{
            std::forward<Args>(args)...
        }.runFakeQuantize();
    }
};


template <typename T>
inline InferenceEngine::Blob::Ptr fp32_to_precision_blob(InferenceEngine::Blob::Ptr fp32_blob, InferenceEngine::Precision precision,
    float scale_factor, const FakeQuantizeParams& fqParams) {
    auto prec_blob = InferenceEngine::make_shared_blob<T>({ precision,
        fp32_blob->getTensorDesc().getDims(), fp32_blob->getTensorDesc().getLayout() });
    prec_blob->allocate();

    auto input_low = 0.0f;
    auto input_high = 0.0f;
    auto output_low = 0.0f;
    auto output_high = 0.0f;
    auto levels = 1;
    if (fqParams.paramsSet) {
        input_low = fqParams.inputMinValue;
        input_high = fqParams.inputMaxValue;
        output_low = fqParams.outputMinValue;
        output_high = fqParams.outputMaxValue;
        levels = fqParams.levelsNum;
    }

    int i = 0;
    for (auto& precValue : *prec_blob) {
        auto f32Value = fp32_blob->buffer().template as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>()[i++];
        if (fqParams.paramsSet) {
            auto x = f32Value;
            if (x <= std::min(input_low, input_high)) {
                f32Value = output_low;
            } else if (x > std::max(input_low, input_high)) {
                f32Value = output_high;
            } else {
                f32Value = nearbyint((x - input_low) / (input_high - input_low) * (levels - 1)) /
                    (levels - 1) * (output_high - output_low) + output_low;
            }
        }

        f32Value = f32Value * scale_factor;
        if (f32Value > std::numeric_limits<T>::max()) {
            precValue = std::numeric_limits<T>::max();
        } else if (f32Value < std::numeric_limits<T>::min()) {
            precValue = std::numeric_limits<T>::min();
        } else {
            precValue = static_cast<T>(f32Value);
        }
    }

    return  static_cast<InferenceEngine::Blob::Ptr>(prec_blob);
}

inline InferenceEngine::Blob::Ptr fp32_to_precision_blob(InferenceEngine::Blob::Ptr fp32_blob, InferenceEngine::Precision precision,
    float scale_factor, const FakeQuantizeParams &fqParams) {
    InferenceEngine::Blob::Ptr result_ptr = nullptr;
    switch (precision) {
    case InferenceEngine::Precision::FP32:
        result_ptr = fp32_to_precision_blob<float>(fp32_blob, precision, scale_factor, fqParams);
        break;
    case InferenceEngine::Precision::I32:
        result_ptr = fp32_to_precision_blob<int32_t>(fp32_blob, precision, scale_factor, fqParams);
        break;
    case InferenceEngine::Precision::I16:
        result_ptr = fp32_to_precision_blob<int16_t>(fp32_blob, precision, scale_factor, fqParams);
        break;
    case InferenceEngine::Precision::I8:
        result_ptr = fp32_to_precision_blob<int8_t>(fp32_blob, precision, scale_factor, fqParams);
        break;
    default:
        THROW_GNA_EXCEPTION << "FP32 to " << precision << " not supported";
    }
    return result_ptr;
}

template <class T, class... Args>
InferenceEngine::Blob::Ptr make_custom_blob(Args&&... args) {
    return InferenceEngine::make_shared_blob<T>(InferenceEngine::Precision::fromType<T>(), std::forward<Args>(args)...);
}

template <class T>
InferenceEngine::Blob::Ptr make_custom_blob(InferenceEngine::Layout layout, InferenceEngine::SizeVector size) {
    return InferenceEngine::make_shared_blob<T>(
        InferenceEngine::TensorDesc(InferenceEngine::Precision::fromType<T>(), size, layout));
}

template<class QuantDesc, class QuantFunc>
inline void quantizeWeightsBiases(const QuantDesc & quantDesc,
                                  InferenceEngine::WeightableLayer *wl,
                                  const QuantFunc &fnc,
                                  bool isDiagonal = false) {  // for diagonal layer number of weights and biases significatly smaller
    // for quantized weights
    auto intWeights =
        make_custom_blob<typename QuantDesc::WeightsPrecision>(InferenceEngine::C, InferenceEngine::SizeVector({wl->_weights->size()}));
    intWeights->allocate();
    if (intWeights->buffer() == nullptr) {
        IE_THROW(NotAllocated)
                << "[GNAPlugin] in function " << __PRETTY_FUNCTION__<< ": "
                << "cannot copy weights for layer :"<< wl->name << " of size" << intWeights->byteSize();
    }

    int oIdx = wl->outData[0]->getDims().size() - 1;
    int iIdx = wl->insData[0].lock().get()->getDims().size() - 1;

    auto getBiasSizeForLayer = [&oIdx](InferenceEngine::WeightableLayer *wl) {
        if (wl->_biases) {
            return wl->_biases->size();
        }
        // calculating biases len using outdata dims
        auto & dims = wl->outData.front()->getDims();
        return dims[oIdx];
    };

    using BiasesPrecision = typename QuantDesc::BiasesPrecision;
    auto biasMaker = [&] () {
        InferenceEngine::Blob::Ptr zero;
        if (!wl->_biases && !shouldAlwaysAllocate<BiasesPrecision>()) {
            return zero;
        }
        auto bias = make_custom_blob<BiasesPrecision>(InferenceEngine::C, InferenceEngine::SizeVector({
            getBiasSizeForLayer(wl)
        }));
        bias->allocate();
        if (bias->buffer() == nullptr) {
            IE_THROW(NotAllocated)
                << "[GNAPlugin] in function " << __PRETTY_FUNCTION__<< ": "
                << "cannot copy bias for layer :"<< wl->name <<"of size" << bias->byteSize();
        }

        memset(bias->buffer(), 0, bias->byteSize());

        return bias;
    };
    auto intBiases = biasMaker();

    float input_scale_factor = 1.f;
    if (InferenceEngine::CNNNetHasPrevLayer(wl)) {
        auto quantDataForInputLayer =
            InferenceEngine::getInjectedData<QuantizedLayerParams>(*InferenceEngine::CNNNetPrevLayer(wl).get());
        input_scale_factor = quantDataForInputLayer->_dst_quant.GetScale();
        if (std::isnan(input_scale_factor) ||
            std::isinf(input_scale_factor)) {
            IE_THROW() << "Unsupported input scale factor value " << input_scale_factor;
        }
    }

    uint32_t num_rows = isDiagonal ? 1 : wl->outData[0]->getDims()[oIdx];
    uint32_t num_columns = isDiagonal ? wl->_weights->size() : wl->insData[0].lock().get()->getDims()[iIdx];

    if (LayerInfo(wl).isAffineFilter() || LayerInfo(wl).isConcatAlignFilter())  {
        // for affine filter layer insdata size not equal to actual coded in input layer
        num_columns = wl->_weights->size() / num_rows;
    }

    if (isDiagonal) {
        std::swap(num_rows, num_columns);
    }

    uint32_t num_rows_padded = num_rows;
    uint32_t num_columns_padded = num_columns;

    // TODO: replace this into fixed scale quantizer then

    auto quantData = InferenceEngine::getInjectedData<QuantizedLayerParams>(*wl);
    {
        auto weightsStats = !quantData->_weights_quant.GetMinValues().empty();
        auto weightsScale = quantData->_weights_quant.GetScale();
        auto dstScale = quantData->_dst_quant.GetScale();
        auto blob_precision = wl->_weights->getTensorDesc().getPrecision();
        auto quantizedWeights = blob_precision != InferenceEngine::Precision::FP32 && blob_precision != InferenceEngine::Precision::FP16;
        fnc(wl->_weights->buffer().as<float*>(),
            wl->_biases ? wl->_biases->buffer().as<float*>() : nullptr,
            intWeights->buffer(),
            intBiases ? intBiases->buffer() : static_cast<BiasesPrecision*>(nullptr),
            input_scale_factor,
            &weightsScale,
            &dstScale,
            num_rows,
            num_columns,
            num_rows_padded,
            num_columns_padded,
            quantizedWeights,
            quantData->_weights_quant.GetLevels(),
            quantData->_weights_quant.GetMinValues().size(),
            weightsStats ? &quantData->_weights_quant.GetMinValues(true).front() : nullptr,
            weightsStats ? &quantData->_weights_quant.GetMaxValues(true).front() : nullptr,
            weightsStats ? &quantData->_weights_quant.GetMinValues(false).front() : nullptr,
            weightsStats ? &quantData->_weights_quant.GetMaxValues(false).front() : nullptr);
    }
    wl->_weights = intWeights;
    wl->_biases = intBiases;

    /**
     * correcting precision for outdata
     */
    wl->precision = quantDesc.getWeightsPrecision();
    for (auto &&outData : wl->outData) {
        outData->setPrecision(quantDesc.getOutputPrecision());
    }
}


template<class QuantDesc, class QuantFunc>
inline void quantizeWeightsBiasesConv(const QuantDesc & quantDesc,
                                  InferenceEngine::WeightableLayer *conv,
                                  const QuantFunc &fnc) {
    // for quantized weights
    auto intWeights = make_custom_blob<typename QuantDesc::WeightsPrecision>(InferenceEngine::C, InferenceEngine::SizeVector({conv->_weights->size()}));
    intWeights->allocate();
    if (intWeights->buffer() == nullptr) {
        IE_THROW(NotAllocated)
            << "[GNAPlugin] in function " << __PRETTY_FUNCTION__<< ": "
            << "cannot copy weights for layer :"<< conv->name << " of size" << intWeights->byteSize();
    }

    auto getBiasSizeForLayer = [](InferenceEngine::WeightableLayer *wl) -> size_t {
        if (wl->_biases) {
            return wl->_biases->size();
        }
        // calculating biases len using outdata dims: biases number should be equal to output channels number
        return InferenceEngine::GetDataDimSize(wl->outData.front(), InferenceEngine::DataDimName::C);
    };

    using BiasesPrecision = typename QuantDesc::BiasesPrecision;
    auto biasMaker = [&] () {
        InferenceEngine::Blob::Ptr zero;
        if (!conv->_biases && !shouldAlwaysAllocate<BiasesPrecision>()) {
            return zero;
        }
        auto bias = make_custom_blob<BiasesPrecision>(InferenceEngine::C, InferenceEngine::SizeVector({
                                                                                                          getBiasSizeForLayer(conv)
                                                                                                      }));
        bias->allocate();
        if (bias->buffer() == nullptr) {
            IE_THROW(NotAllocated)
                << "[GNAPlugin] in function " << __PRETTY_FUNCTION__<< ": "
                << "cannot copy bias for layer :"<< conv->name <<"of size" << bias->byteSize();
        }
        memset(bias->buffer(), 0, bias->byteSize());

        return bias;
    };
    auto intBiases = biasMaker();

    float input_scale_factor = 1.f;
    if (InferenceEngine::CNNNetHasPrevLayer(conv)) {
        auto quantDataForInputLayer =
            InferenceEngine::getInjectedData<QuantizedLayerParams>(*InferenceEngine::CNNNetPrevLayer(conv).get());
        input_scale_factor = quantDataForInputLayer->_dst_quant.GetScale();
        if (std::isnan(input_scale_factor) ||
            std::isinf(input_scale_factor)) {
            IE_THROW() << "Unsupported input scale factor value " << input_scale_factor;
        }
    }
    if (conv->outData[0]->getDims().size() < 2) {
        IE_THROW() << "Unsupported output dims size for " << conv->name <<", should be > 1, but " << conv->outData[0]->getDims().size();
    }
    if (conv->insData[0].lock().get()->getDims().size() < 2) {
        IE_THROW() << "Unsupported input dims size for " << conv->name << ", should be > 1, but " << conv->insData[0].lock().get()->getDims().size();
    }
    auto inputData = conv->insData[0].lock();

    uint32_t num_rows = getBiasSizeForLayer(conv);
    if (num_rows == 0) {
        THROW_GNA_EXCEPTION << "Invalid num rows";
    }
    uint32_t num_columns = conv->_weights->size() / num_rows;

    uint32_t num_rows_padded = num_rows;
    uint32_t num_columns_padded = num_columns;

    // TODO: replace this into fixed scale quantizer then

    auto quantData = InferenceEngine::getInjectedData<QuantizedLayerParams>(*conv);
    {
        auto weightsStats = !quantData->_weights_quant.GetMinValues().empty();
        auto weightsScale = quantData->_weights_quant.GetScale();
        auto dstScale = quantData->_dst_quant.GetScale();
        auto blob_precision = conv->_weights->getTensorDesc().getPrecision();
        auto quantizedWeights = blob_precision != InferenceEngine::Precision::FP32 && blob_precision != InferenceEngine::Precision::FP16;
        fnc(conv->_weights->buffer().as<float*>(),
            conv->_biases ? conv->_biases->buffer().as<float*>() : nullptr,
            intWeights->buffer(),
            intBiases ? intBiases->buffer() : static_cast<BiasesPrecision*>(nullptr),
            input_scale_factor,
            &weightsScale,
            &dstScale,
            num_rows,
            num_columns,
            num_rows_padded,
            num_columns_padded,
            quantizedWeights,
            quantData->_weights_quant.GetLevels(),
            quantData->_weights_quant.GetMinValues().size(),
            weightsStats ? &quantData->_weights_quant.GetMinValues(true).front() : nullptr,
            weightsStats ? &quantData->_weights_quant.GetMaxValues(true).front() : nullptr,
            weightsStats ? &quantData->_weights_quant.GetMinValues(false).front() : nullptr,
            weightsStats ? &quantData->_weights_quant.GetMaxValues(false).front() : nullptr);
    }
    conv->_weights = intWeights;
    conv->_biases = intBiases;

    /**
     * correcting precision for outdata
     */
    conv->precision = quantDesc.getWeightsPrecision();
    for (auto &&outData : conv->outData) {
        outData->setPrecision(quantDesc.getOutputPrecision());
    }
}


class DataQuantizerBase {
 public:
    explicit DataQuantizerBase(float scaleFactor) : scaleFactor(scaleFactor) {
    }
 protected:
    float scaleFactor = 1.0;
};
/**
 * Helper class to use partial specialisation of Layer type
 * @tparam Desc
 * @tparam Layer
 */
template<class Desc, class Layer>
class DataQuantizer : public DataQuantizerBase {
 public:
    explicit DataQuantizer(float scaleFactor) : DataQuantizerBase(scaleFactor) {}
    bool operator()(Layer cnnLayer) const {
        return false;
    }
};

template<class Desc>
class DataQuantizer<Desc, InferenceEngine::CNNLayer *> : public DataQuantizerBase {
 public:
    explicit DataQuantizer(float scaleFactor) : DataQuantizerBase(scaleFactor) {}

    bool operator()(InferenceEngine::CNNLayer *cnnLayer) const {
        for (auto &&outData : cnnLayer->outData) {
            outData->setPrecision(Desc::mandatory().getOutputPrecision());
        }
        // set scale factor for input layers
        if (cnnLayer->insData.empty()) {
            for (auto &&outData : cnnLayer->outData) {
                outData->setPrecision(Desc::mandatory().getInputPrecision());
            }
        } else {
            if (LayerInfo(*cnnLayer).isActivation() ||
                    LayerInfo(*cnnLayer).isCopy() ||
                    LayerInfo(*cnnLayer).isNonFunctional() ||
                    LayerInfo(*cnnLayer).isPermute() ||
                    LayerInfo(*cnnLayer).isConst()) {
                // precision of activation layers is always equal input precision
                for (auto &&outData : cnnLayer->outData) {
                    outData->setPrecision(Desc::mandatory().getInputPrecision());
                }
            }
            // for pooling layer output precision is the same as input precision
            if (LayerInfo(*cnnLayer).isMaxPooling()) {
                const auto inputPrecision = cnnLayer->insData.front().lock()->getPrecision();
                for (auto&& outData : cnnLayer->outData) {
                    outData->setPrecision(inputPrecision);
                }
            }
        }
        cnnLayer->precision = Desc::mandatory().getInputPrecision();

        if (LayerInfo(*cnnLayer).isConst()) {
            auto initial_precision = cnnLayer->blobs["custom"]->getTensorDesc().getPrecision();
            // TODO I32 must be handled separately when it'll be supported
            IE_ASSERT(initial_precision != InferenceEngine::Precision::I32);

            if (initial_precision == InferenceEngine::Precision::FP16) {
                cnnLayer->blobs["custom"] = make_fp32_blob(cnnLayer->blobs["custom"]);
            }
            auto quantParams = InferenceEngine::getInjectedData<QuantizedLayerParams>(*cnnLayer);
            auto new_const_blob = InferenceEngine::Blob::CreateFromData(cnnLayer->outData[0]);
            auto const_blob = cnnLayer->blobs["custom"];
            if (const_blob->getTensorDesc().getPrecision() == InferenceEngine::Precision::FP32) {
                auto fqParams = FakeQuantizeParams{};
                if (quantParams->_dst_quant.IsStatsSet()) {
                    fqParams.paramsSet = true;
                    fqParams.levelsNum = quantParams->_dst_quant.GetLevels();
                    fqParams.inputMinValue = quantParams->_dst_quant.GetMinValues(true).front();
                    fqParams.inputMaxValue = quantParams->_dst_quant.GetMaxValues(true).front();
                    fqParams.outputMinValue = quantParams->_dst_quant.GetMinValues(false).front();
                    fqParams.outputMaxValue = quantParams->_dst_quant.GetMaxValues(false).front();
                }

                cnnLayer->blobs["custom"] = fp32_to_precision_blob(const_blob, cnnLayer->outData[0]->getPrecision(),
                    quantParams->_dst_quant.GetScale(), fqParams);
            }
        }

        return true;
    }
};


template<class Desc>
class DataQuantizer<Desc, InferenceEngine::SplitLayer *> : public DataQuantizer<Desc, InferenceEngine::CNNLayer *> {
    using base = DataQuantizer<Desc, InferenceEngine::CNNLayer *>;
 public:
    explicit DataQuantizer(float scaleFactor) : base(scaleFactor) {}
    bool operator()(InferenceEngine::SplitLayer *splitLayer) const {
        base::operator()(splitLayer);
        // split layer doesnt change it's data at all
        for (auto &&outData : splitLayer->outData) {
            outData->setPrecision(Desc::mandatory().getInputPrecision());
        }
        return true;
    }
};

template<class Desc>
class DataQuantizer<Desc, InferenceEngine::ConcatLayer *> : public DataQuantizer<Desc, InferenceEngine::CNNLayer *> {
    using base = DataQuantizer<Desc, InferenceEngine::CNNLayer *>;
 public:
    explicit DataQuantizer(float scaleFactor) : base(scaleFactor) {}
    bool operator()(InferenceEngine::ConcatLayer *concatLayer) const {
        base::operator()(concatLayer);
        for (auto &&outData : concatLayer->outData) {
            outData->setPrecision(Desc::mandatory().getInputPrecision());
        }
        return true;
    }
};

template<class Desc>
class DataQuantizer<Desc, InferenceEngine::CropLayer *> : public DataQuantizer<Desc, InferenceEngine::CNNLayer *> {
    using base = DataQuantizer<Desc, InferenceEngine::CNNLayer *>;
 public:
    explicit DataQuantizer(float scaleFactor) : base(scaleFactor) {}
    bool operator()(InferenceEngine::CropLayer *cropLayer) const {
        base::operator()(cropLayer);
        for (auto &&outData : cropLayer->outData) {
            outData->setPrecision(Desc::mandatory().getInputPrecision());
        }
        return true;
    }
};

template<class Desc>
class DataQuantizer<Desc, InferenceEngine::ReshapeLayer *> : public DataQuantizer<Desc, InferenceEngine::CNNLayer *> {
    using base = DataQuantizer<Desc, InferenceEngine::CNNLayer *>;
 public:
    explicit DataQuantizer(float scaleFactor) : base(scaleFactor) {}
    bool operator()(InferenceEngine::ReshapeLayer *reshapeLayer) const {
        base::operator()(reshapeLayer);
        // reshape layer doesnt change it's data at all
        for (auto &&outData : reshapeLayer->outData) {
            outData->setPrecision(reshapeLayer->insData.front().lock()->getPrecision());
        }
        return true;
    }
};

template<class Desc>
class DataQuantizer<Desc, InferenceEngine::WeightableLayer *> : public DataQuantizerBase {
 public:
    explicit DataQuantizer(float scaleFactor) : DataQuantizerBase(scaleFactor) {}
    bool operator()(InferenceEngine::WeightableLayer *wl) const {
        quantizeWeightsBiases<typename Desc::MandatoryType>(Desc::mandatory(), wl, Quant<typename Desc::MandatoryType>());
        return true;
    }
};

template<class Desc>
class DataQuantizer<Desc, InferenceEngine::ConvolutionLayer *> : public DataQuantizerBase {
 public:
    explicit DataQuantizer(float scaleFactor) : DataQuantizerBase(scaleFactor) {}
    bool operator()(InferenceEngine::ConvolutionLayer *cl) const {
        quantizeWeightsBiasesConv<typename Desc::OptionalType>(Desc::optional(), cl, Quant<typename Desc::OptionalType>());
        return true;
    }
};

template<class Desc>
class DataQuantizer<Desc, InferenceEngine::ScaleShiftLayer *> : public DataQuantizerBase {
 public:
    explicit DataQuantizer(float scaleFactor) : DataQuantizerBase(scaleFactor) {}
    bool operator()(InferenceEngine::ScaleShiftLayer *ssl) const {
        quantizeWeightsBiases<typename Desc::OptionalType>(Desc::optional(), ssl, Quant<typename Desc::OptionalType>(), true);
        return true;
    }
};

}  // namespace frontend

template<class Desc>
class LayersQuantizer : public frontend::DataQuantizerBase {
 public:
    explicit LayersQuantizer(float scaleFactor) : DataQuantizerBase(scaleFactor) {}
    template<class T>
    bool operator()(T input) const {
        return frontend::DataQuantizer<Desc, T>(scaleFactor)(input);
    }
};

using QuantI16 = frontend::QuantPair<frontend::QuantI16, frontend::QuantI16>;
using QuantI8 = frontend::QuantPair<frontend::QuantI8, frontend::QuantI16>;
using QuantI8_I8 = frontend::QuantPair<frontend::QuantI8_I8, frontend::QuantI8_I8>;


using FakeQuantI16 = frontend::QuantPair<frontend::FakeQuantI16, frontend::FakeQuantI16>;
using FakeQuantI8 = frontend::QuantPair<frontend::FakeQuantI8, frontend::FakeQuantI16>;


}  // namespace GNAPluginNS
