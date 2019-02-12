// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <tuple>
#include <utility>

#include "ie_layers.h"

namespace InferenceEngine {

namespace details {

template <class T, class InjectType>
class LayerInjector : public T {
public:
    InjectType injected;
    explicit LayerInjector(const T& base): T(base) {}
};

using AllLayers =
    std::tuple<SelectLayer*, DeformableConvolutionLayer*, DeconvolutionLayer*, ConvolutionLayer*, TopKLayer*,
               PoolingLayer*, FullyConnectedLayer*, GemmLayer*, PadLayer*, GatherLayer*, StridedSliceLayer*,
               ShuffleChannelsLayer*, DepthToSpaceLayer*, SpaceToDepthLayer*, SparseFillEmptyRowsLayer*,
               SparseSegmentReduceLayer*, ExperimentalSparseWeightedReduceLayer*, SparseToDenseLayer*, BucketizeLayer*,
               ReverseSequenceLayer*, RangeLayer*, FillLayer*, BroadcastLayer*, ConcatLayer*,
               SplitLayer*, NormLayer*, SoftMaxLayer*, GRNLayer*, MVNLayer*, ReLULayer*, EltwiseLayer*, CropLayer*,
               ReshapeLayer*, TileLayer*, ScaleShiftLayer*, PReLULayer*, PowerLayer*, BatchNormalizationLayer*,
               ClampLayer*, TensorIterator*, LSTMCell*, GRUCell*, RNNCell*, RNNSequenceLayer*, QuantizeLayer*,
               BinaryConvolutionLayer*, WeightableLayer*, OneHotLayer*, MathLayer*, ReduceLayer*, UniqueLayer*,
               NonMaxSuppressionLayer*, ScatterLayer*, TanHLayer*, SigmoidLayer*, CNNLayer*>;

/**
 * @brief checks whether type inxed as P has a parent among element in range I..N
 * can be used only for P < I
 * */
template <size_t P, size_t I, class Tuple, class Enable = void>
struct is_base_of_any;

// clang-format off
template <size_t IBase, size_t IDerived, class Tuple>
struct is_base_of_any< IBase, IDerived, Tuple,
    typename std::enable_if<IBase < std::tuple_size<Tuple>::value, void>::type > : public std::true_type {
         using base = typename std::remove_pointer<typename std::tuple_element<IBase, Tuple>::type>::type;
         using derived = typename std::remove_pointer<typename std::tuple_element<IDerived, Tuple>::type>::type;

    static_assert(IDerived < IBase, "cannot match parent using incorrect indices");
    static_assert(!std::is_base_of<derived, base>::value, "probing type is a parent of followed type");

    // check that incoming type have parents in range I..N, and any of I..N not a child of derived type
     static_assert((std::is_base_of<base, derived>::value || is_base_of_any<IBase + 1, IDerived, Tuple>::value), "parent matching failed");
};
// clang-format on

// for matches any->after last
template <size_t IBase, size_t IDerived, class Tuple>
struct is_base_of_any<IBase, IDerived, Tuple,
                      typename std::enable_if<IBase >= std::tuple_size<Tuple>::value, void>::type>
    : public std::false_type {};

/**
 * @brief check whether type ordered from child to base within given list
 */
template <size_t P, class Tuple, class Enable = void>
struct is_types_ordered_from_child_to_base {};

template <size_t P, class Tuple>
struct is_types_ordered_from_child_to_base<
    P, Tuple, typename std::enable_if<P != std::tuple_size<Tuple>::value - 2, void>::type> {
    static constexpr bool value =
        is_base_of_any<P + 1, P, Tuple>::value && is_types_ordered_from_child_to_base<P + 1, Tuple>::value;
};

template <size_t P, class Tuple>
struct is_types_ordered_from_child_to_base<
    P, Tuple, typename std::enable_if<P == std::tuple_size<Tuple>::value - 2, void>::type> {
    static constexpr bool value = is_base_of_any<P + 1, P, Tuple>::value;
};

static_assert(
    is_types_ordered_from_child_to_base<0, AllLayers>::value,
    "All layers must be topologically sorted as so for any layer, it's father appeared later in a types list");

template <typename InjectedType, typename T>
inline void dynamic_cast_layer(const CNNLayer& source, CNNLayerPtr& target, T& /*, InjectedType value*/) {
    if (target) {
        return;
    }
    auto casted = dynamic_cast<const T*>(source);
    if (casted != nullptr) {
        auto layerWithInjectedData = std::make_shared<LayerInjector<T, InjectedType>>(
            LayerParams {casted->name, casted->type, casted->precision});
        target = layerWithInjectedData;
    }
}

template <class Visitor, std::size_t I = 0, typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), void>::type visitActualLayer(std::tuple<Tp...>&& t,
                                                                                const CNNLayer& sourceLayer,
                                                                                const Visitor& v) {}

template <class Visitor, std::size_t I = 0, typename... Tp>
    inline typename std::enable_if <
    I<sizeof...(Tp), void>::type visitActualLayer(std::tuple<Tp...>&& t, const CNNLayer& sourceLayer,
                                                  const Visitor& visitor) {
    using EType = typename std::tuple_element<I, std::tuple<Tp...>>::type;
    auto casted = dynamic_cast<EType>(const_cast<CNNLayer*>(&sourceLayer));

    if (casted != nullptr) {
        // means no need to handle further layers
        if (visitor(casted)) {
            return;
        }
    }

    visitActualLayer<Visitor, I + 1, Tp...>(std::move(t), sourceLayer, visitor);
}

template <class InjectedType, std::size_t I = 0, typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), void>::type injectHelper(std::tuple<Tp...>& t,
                                                                            const CNNLayer& sourceLayer,
                                                                            CNNLayerPtr& targetLayer,
                                                                            const InjectedType& value) {}

template <class InjectedType, std::size_t I = 0, typename... Tp>
    inline typename std::enable_if <
    I<sizeof...(Tp), void>::type injectHelper(std::tuple<Tp...>& t, const CNNLayer& sourceLayer, CNNLayerPtr& target,
                                              const InjectedType& value) {
    if (target) {
        return;
    }
    using EType = typename std::tuple_element<I, std::tuple<Tp...>>::type;
    auto casted = dynamic_cast<EType>(const_cast<CNNLayer*>(&sourceLayer));

    if (casted != nullptr) {
        auto layerWithInjectedData =
            std::make_shared<LayerInjector<typename std::remove_pointer<EType>::type, InjectedType>>(*casted);

        // copy outdata
        for (auto&& data : layerWithInjectedData->outData) {
            data = std::make_shared<Data>(*data.get());
        }

        layerWithInjectedData->injected = value;

        target = layerWithInjectedData;
    }

    injectHelper<InjectedType, I + 1, Tp...>(t, sourceLayer, target, value);
}

template <class InjectedType, std::size_t I = 0, typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), void>::type locateInjected(std::tuple<Tp...>& t,
                                                                              const CNNLayer& sourceLayer,
                                                                              InjectedType*& value) {}

template <class InjectedType, std::size_t I = 0, typename... Tp>
    inline typename std::enable_if <
    I<sizeof...(Tp), void>::type locateInjected(std::tuple<Tp...>& t, const CNNLayer& sourceLayer,
                                                InjectedType*& value) {
    if (value != nullptr) {
        return;
    }
    using EType = typename std::tuple_element<I, std::tuple<Tp...>>::type;
    auto injectedLayer = dynamic_cast<LayerInjector<typename std::remove_pointer<EType>::type, InjectedType>*>(
        const_cast<CNNLayer*>(&sourceLayer));

    if (injectedLayer != nullptr) {
        value = &injectedLayer->injected;
    }

    locateInjected<InjectedType, I + 1, Tp...>(t, sourceLayer, value);
}

}  // namespace details

/**
 * @brief creates copy of source layer, with injected arbitrary data
 * @tparam InjectType data type to be injected
 * @param sourceLayer
 * @param value injected value
 * @return newly created layer with injected data
 */
template <class InjectType>
inline CNNLayerPtr injectData(const CNNLayer& sourceLayer, const InjectType& value = InjectType()) {
    details::AllLayers layers;
    CNNLayerPtr targetLayer;
    details::injectHelper(layers, sourceLayer, targetLayer, value);

    return targetLayer;
}

template <class InjectType>
inline CNNLayerPtr injectData(CNNLayerPtr sourceLayer, const InjectType& value = InjectType()) {
    return injectData(*sourceLayer.get(), value);
}

/**
 * @brief transforms of source layer
 * @tparam InjectType data type to be injected
 * @param sourceLayer
 * @param value injected value
 * @return newly created layer with injected data
 */
template <class Transformer>
inline void transformLayer(const CNNLayer& sourceLayer, const Transformer& transformer) {
    details::visitActualLayer<Transformer>(std::move(details::AllLayers()), sourceLayer, transformer);
}

template <class Transformer>
inline void transformLayer(CNNLayerPtr sourceLayer, const Transformer& transformer) {
    transformLayer(*sourceLayer.get(), transformer);
}

/**
 * @brief  getPointer to injected data
 * @tparam InjectType
 * @param sourceLayer
 * @return if previously data of type InjectType was injected, will return pointer to it, nullptr otherwise
 */
template <class InjectType>
inline InjectType* getInjectedData(const CNNLayer& sourceLayer) {
    details::AllLayers layers;
    InjectType* injected = nullptr;

    details::locateInjected(layers, sourceLayer, injected);

    return injected;
}

template <class InjectType>
inline InjectType* getInjectedData(CNNLayerPtr sourceLayer) {
    return getInjectedData<InjectType>(*sourceLayer.get());
}

}  // namespace InferenceEngine
