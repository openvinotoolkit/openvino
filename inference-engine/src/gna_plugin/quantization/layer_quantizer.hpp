// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <string>
#include <utility>
#include <gna-api-types-xnn.h>
#include "ie_layers.h"
#include "quantized_layer_params.hpp"
#include "quantization.h"
#include "details/caseless.hpp"
#include "graph_tools.hpp"
#include "blob_factory.hpp"
#include "precision_ex.hpp"
#include "pwl.h"
#include "gna_layer_info.hpp"

namespace GNAPluginNS {
namespace details {

/**
 * @brief description of quantisation precision
 * @tparam Ip - input precision
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
struct QuantI8  : public QuantDescTmpl<P_TYPE(I16), P_TYPE(I32), P_TYPE(I8), intel_compound_bias_t, P_TYPE(MIXED)> {
    QuantI8() {
        _Np = InferenceEngine::Precision::MIXED;
    }
};

template <class A, class B>
struct QuantPair {
    using MandatoryType = A;
    using OptionalType = B;
    static A mandatory () { return A();}
    static B optional () { return B();}
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
inline bool shouldAlwaysAllocate<intel_compound_bias_t>() {
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
        QuantizeAffine16(std::forward<Args>(args)...);
    }
};

template<>
class Quant<QuantI8> {
 public:
    template<class ...Args>
    void operator()(Args && ... args) const {
        QuantizeAffine8(std::forward<Args>(args)...);
    }
};

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
        THROW_GNA_EXCEPTION << InferenceEngine::details::as_status << InferenceEngine::NOT_ALLOCATED
                            << "cannot copy weights for layer :"<< wl->name << " of size" << intWeights->byteSize();
    }


    auto getBiasSizeForLayer = [](InferenceEngine::WeightableLayer *wl) {
        if (wl->_biases) {
            return wl->_biases->size();
        }
        // calculating biases len using weight dims
        auto & dims = wl->outData.front()->getDims();
        return dims[1];
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
            THROW_GNA_EXCEPTION << InferenceEngine::details::as_status << InferenceEngine::NOT_ALLOCATED
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
        input_scale_factor = quantDataForInputLayer->_dst_quant.scale;
        if (std::isnan(input_scale_factor) ||
            std::isinf(input_scale_factor)) {
            THROW_IE_EXCEPTION << "Unsupported input scale factor value " << input_scale_factor;
        }
    }
    if (wl->outData[0]->getDims().size() < 2) {
        THROW_IE_EXCEPTION << "Unsupported output dims size for " << wl->name <<", should be > 1, but " << wl->outData[0]->getDims().size();
    }
    if (wl->insData[0].lock().get()->getDims().size() < 2) {
        THROW_IE_EXCEPTION << "Unsupported input dims size for " << wl->name << ", should be > 1, but " << wl->insData[0].lock().get()->getDims().size();
    }
    uint32_t num_rows = isDiagonal ? 1 : wl->outData[0]->getDims()[1];
    uint32_t num_columns = wl->insData[0].lock().get()->getDims()[1];

    if (isDiagonal) {
        std::swap(num_rows, num_columns);
    }

    uint32_t num_rows_padded = num_rows;
    uint32_t num_columns_padded = num_columns;

    // TODO: replace this into fixed scale quantizer then

    auto quantData = InferenceEngine::getInjectedData<QuantizedLayerParams>(*wl);
    {
        fnc(wl->_weights->buffer().as<float *>(),
            wl->_biases ? wl->_biases->buffer().as<float *>() : nullptr,
            intWeights->buffer(),
            intBiases ? intBiases->buffer() : static_cast<BiasesPrecision *>(nullptr),
            input_scale_factor,
            &quantData->_weights_quant.scale,
            &quantData->_dst_quant.scale,
            num_rows,
            num_columns,
            num_rows_padded,
            num_columns_padded);
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
        THROW_GNA_EXCEPTION << InferenceEngine::details::as_status << InferenceEngine::NOT_ALLOCATED
                            << "cannot copy weights for layer :"<< conv->name << " of size" << intWeights->byteSize();
    }


    auto getBiasSizeForLayer = [](InferenceEngine::WeightableLayer *wl) {
        if (wl->_biases) {
            return wl->_biases->size();
        }
        // calculating biases len using weight dims
        auto & dims = wl->outData.front()->getDims();
        return dims[1];
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
            THROW_GNA_EXCEPTION << InferenceEngine::details::as_status << InferenceEngine::NOT_ALLOCATED
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
        input_scale_factor = quantDataForInputLayer->_dst_quant.scale;
        if (std::isnan(input_scale_factor) ||
            std::isinf(input_scale_factor)) {
            THROW_IE_EXCEPTION << "Unsupported input scale factor value " << input_scale_factor;
        }
    }
    if (conv->outData[0]->getDims().size() < 2) {
        THROW_IE_EXCEPTION << "Unsupported output dims size for " << conv->name <<", should be > 1, but " << conv->outData[0]->getDims().size();
    }
    if (conv->insData[0].lock().get()->getDims().size() < 2) {
        THROW_IE_EXCEPTION << "Unsupported input dims size for " << conv->name << ", should be > 1, but " << conv->insData[0].lock().get()->getDims().size();
    }
    auto inputData = conv->insData[0].lock();

    uint32_t num_rows = getBiasSizeForLayer(conv);
    uint32_t num_columns = conv->_weights->size() / num_rows;

    uint32_t num_rows_padded = num_rows;
    uint32_t num_columns_padded = num_columns;

    // TODO: replace this into fixed scale quantizer then

    auto quantData = InferenceEngine::getInjectedData<QuantizedLayerParams>(*conv);
    {
        fnc(conv->_weights->buffer().as<float *>(),
            conv->_biases ? conv->_biases->buffer().as<float *>() : nullptr,
            intWeights->buffer(),
            intBiases ? intBiases->buffer() : static_cast<BiasesPrecision *>(nullptr),
            input_scale_factor,
            &quantData->_weights_quant.scale,
            &quantData->_dst_quant.scale,
            num_rows,
            num_columns,
            num_rows_padded,
            num_columns_padded);
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
        auto quantData = InferenceEngine::getInjectedData<QuantizedLayerParams>(*cnnLayer);
        if (cnnLayer->insData.empty()) {
            for (auto &&outData : cnnLayer->outData) {
                outData->setPrecision(Desc::mandatory().getInputPrecision());
            }
        } else {
                if (LayerInfo(*cnnLayer).isActivation() ||
                        LayerInfo(*cnnLayer).isCopy()) {
                // precision of activation layers is always equal input precision
                for (auto &&outData : cnnLayer->outData) {
                    outData->setPrecision(Desc::mandatory().getInputPrecision());
                }
            }
        }
        cnnLayer->precision = Desc::mandatory().getInputPrecision();

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
            outData->setPrecision(Desc::mandatory().getInputPrecision());
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
    bool operator()(InferenceEngine::WeightableLayer *wl) const {
        quantizeWeightsBiasesConv<typename Desc::OptionalType>(Desc::optional(), wl, Quant<typename Desc::OptionalType>());
        return true;
    }
};

template<class Desc>
class DataQuantizer<Desc, InferenceEngine::ScaleShiftLayer *> : public DataQuantizerBase {
 public:
    explicit DataQuantizer(float scaleFactor) : DataQuantizerBase(scaleFactor) {}
    bool operator()(InferenceEngine::ScaleShiftLayer *wl) const {
        quantizeWeightsBiases<typename Desc::OptionalType>(Desc::optional(), wl, Quant<typename Desc::OptionalType>(), true);
        return true;
    }
};

}  // namespace details

template<class Desc>
class LayersQuantizer : public details::DataQuantizerBase {
 public:
    explicit LayersQuantizer(float scaleFactor) : DataQuantizerBase(scaleFactor) {}
    template<class T>
    bool operator()(T input) const {
        return details::DataQuantizer<Desc, T>(scaleFactor)(input);
    }
};

using QuantI16 = details::QuantPair<details::QuantI16, details::QuantI16>;
using QuantI8 = details::QuantPair<details::QuantI8, details::QuantI16>;

}  // namespace GNAPluginNS
