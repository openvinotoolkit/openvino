// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpp/ie_cnn_network.h>
#include <float.h>

#include <ie_icnn_network.hpp>
#include <ie_icnn_network_stats.hpp>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace InferenceEngine {
namespace details {

/**
 * We have raw statistic from stat collection tool and this statistic should be processed to get best
 * accuracy. This transformation depends on the topology, depends on the parameters of layers.
 * i.e. data going to regular and depth-wise convolution would be scaled differently. In case of
 * regular convolution it should be scaled for tensor wide approach, for depth-wise convolution it
 * should be scaled by channel approach.
 * This class contains logic of getting scales
 */
class CNNStatisticHelper {
public:
    /**
     * We need to have topology to make a decision about scales
     * @param network initial network to be quantized, the topology can be changed during quantization
     * @param internalNodesStats initial statistic
     * @param maxSign - maximal signed value to be used for calculation of scales
     * @param maxUnsign - maximal unsigned value to be used for calculation of scales
     *
     */
    CNNStatisticHelper(CNNNetwork& network, const std::map<std::string, NetworkNodeStatsPtr>& internalNodesStats,
                       int maxSign, int maxUnsign);

    /**
     * Returns if we can quantize layer basing on information of existing statistic before and after
     * layers
     */
    bool canLayerBeQuantized(CNNLayer::Ptr layer) const;

    /**
     * The topology is allowed to be changed, we need to modify statistic accordingly
     *
     * Currently there is a need in copy of statistic only

     * @param srcName name of layer from statistic needs to be taken
     * @param dstName name of layer which statistic will be applied
     */
    void copyStatistics(const std::string& srcName, const std::string& dstName);

    /**
     * Returns boolean values if layer produce negative data according collected statistic
     * true means that layer produices negative values
     * false means that layer produces only positive numbers
     * @param layer - layer of interest
     * @param outputPort - number of port to verify. -1 stands forverification of all outputs from
     * layer
     */
    bool hasNegativeOutput(const std::string& layerName, int outputPort = -1) const;

    /**
     * Returns input scale for layer based on statistic
     * @return blob with scales per channel
     */
    InferenceEngine::Blob::Ptr getInputScale(CNNLayer::Ptr layer) const;

    /**
     * Returns output scale for layer based on statistic
     * @return blob with scales per channel
     */
    InferenceEngine::Blob::Ptr getOutputScale(CNNLayer::Ptr layer) const;

    /**
     * provides max signed value as the only place for synchronization with other algorithms in
     * normalizer which require this
     */
    int getMaxSignValue() const;

    /**
     * Returns a latest layer in fusion, the data from returned layer will go to anopther, this mean
     * that for all layers which will be fused we will have to use only statistic from that latest layer
     * @param layer - layer of interest
     *
     * @return returns layer which statistic should be used for calculatio of all scales for layer
     *         passed as a parameter for this method
     */
    CNNLayer::Ptr getLatestInFuse(CNNLayer::Ptr layer) const;

private:
    /**
     * Calculates scale factor according statistic for layer passed to this function. No other logic for
     * selection another layer is implemented here.
     *
     * @param channels redundant parameter, should be removed
     * @param stats redundant parameter, should be removed
     * @param maxInt - we can quantize to I8 even if data is unsigned, need to provide such max number
     *               explicitly
     *
     * @return InferenceEngine::Blob::Ptr
     */
    InferenceEngine::Blob::Ptr calculateScaleFactor(size_t channels, NetworkNodeStatsPtr stats, int maxInt) const;

    /**
     * Select the latet layer in the fusion and returns its statistic
     */
    NetworkNodeStatsPtr getStatistic(CNNLayer::Ptr layer) const;

    /**
     * Pass over alls statistic and normalize it to the only scale per tenso, individual per channel or
     * mix depenging on the pattern in the network
     */
    void NormalizeStatistic();

    CNNNetwork network_;
    std::map<std::string, NetworkNodeStatsPtr> internalNodesStats_;
    int maxSign_;
    int maxUnsign_;
};

/**
 * This class normalizes and quantizes network to "Int8" state
 * The converted network will have
 *  1) scaleshifts which will normalize activation values to int8 (S8/U8) range
 *  2) quantize weigths and biases of convolution
 *  3) adds special attributes to layers because semantic of int8 layer are different vs floating
 *  point ones. For example, after convolution we need to return back to denormalized values and
 *  there should be special scale here
 *  4) Transforms some layers to another ones. For example if i8 to i8 Scaleshift is not supported
 *  by backend, this scaleshift will be converted to grouped/(depth-wise in ideal case) convolution
 *
 *  This class very depends on backend and its fusion. It assumes that fusion must be executed all
 *  the time, we cannot for split it to independent execution of two layers in int8 mode. This is
 *  done to calculate normalization factors the most optimal way to save accuracy.
 *  Currently supported fusion
 *  1. Conv-ReLU
 *  2. Conv-Sum-ReLU which is appeared from the pattern
 *  Conv        Something
 *    \            /
 *        Eltwise
 *         ReLU
 *  Here, the output form "Something" will be used as in-place storge for accumulation of the
 *  results for convolution. That lead to tricky case in int8 when we have signed int8 input and
 *  unsigned u8 output
 *  */
class INFERENCE_ENGINE_API_CLASS(CNNNetworkInt8Normalizer) {
public:
    CNNNetworkInt8Normalizer() {}

private:
    /** Helper function for filling of scaleshift weights for normalization of activation */
    static void fillInScaleShift(ScaleShiftLayer* scshLayer, size_t c, float* weightsN, float* weightsD);

public:
    /** main function for calling of quantization */
    static void NormalizeNetwork(ICNNNetwork& network, ICNNNetworkStats& netStats);

protected:
    /** Helper function to add scaleshifts and other layers for transformatin of topology */
    static void AddLayerToCNNNetworkBeforeLayer(CNNLayer::Ptr newLayer, CNNLayer::Ptr successor, size_t port);
    /** Helper function to add scaleshifts and other layers for transformatin of topology */
    static void AddLayerToCNNNetworkAfterData(DataPtr pData, CNNLayer::Ptr layer, const std::string& nextLayerName);
    /**  Adds ScaleShift between two specified layers  */
    static void AddScaleShiftBetween(CNNNetwork& net, const CNNLayerPtr layer1, const CNNLayerPtr layer2,
                                     CNNStatisticHelper& statHelper);

    /** creates dw convolution with unary weights and zero biases with i8 output and the same
     *  statistic. it will provide requantization from U8 to I8*/
    static CNNLayer::Ptr addU8ToI8Conversion(DataPtr data, CNNLayer::Ptr successor, CNNStatisticHelper& statHelper);

    /**
     * Function which recalculate weights according to input scales, and quantize weights, biases and
     * adds o-scale and w-scale
     * w-scale - multiplication on this scale of i8 convolution result will produce denormalized fp32
     * data
     * o-scale - multiplication on this scale will convert above denormalized fp32 to i8 for next layer
     */
    static void QuantizeConvolutionOrFullyConnected(CNNLayer::Ptr convolution, CNNStatisticHelper& statHelper);

    /**  Adds ScaleShifts everywhere */
    static void AddScaleShifts(CNNNetwork& net, CNNStatisticHelper& statHelper);

    /**  Convert ReLu-like Clamps to ReLu layers */
    static void ClampsToReLU(CNNNetwork& net, CNNStatisticHelper& statHelper);

    /**
     * Goes over all layers and mark which layers will be executed in FP32/I8 and marks data between
     * layers to I8/U8/FP32
     */
    static void DefinesExecutionPrecision(CNNNetwork& net, CNNStatisticHelper& statHelper);

    /**
     * Since o-scales exist only for convolutins, we need to propagate them down oever concats and
     * linear layers
     */
    static void PropagateScaleFactors(CNNNetwork& net, const CNNStatisticHelper& statHelper);

    /**
     * Normalizes and quantizes srcData using scales for normalization and int8blob precision for
     * quantization
     */
    static void ScaleDataToInt(const float* srcData, size_t srcSize, Blob::Ptr int8blob,
                               const std::vector<float>& scales);

    /**
     * Replaces all ScaleShifts layers met in the model to the depth-wise convolution with the same
     * weights and biases.
     *
     * Exceptions:
     * 1. ScaleShift following after Input layer, it is not converted to depth-wise convolution
     * 2. Scaleshift producing output of network
     * 3. Scaleshift passing data to Priorbox
     *
     * This conversion allows to avoid introductin one more i8 primitive - ScaleShift accepting i8 input
     * and producing i8 output
     */
    static void replaceScaleShiftByDWConvolution(CNNNetwork& net);

    /** Helper function which creates DW/Grouped/regular convolution by passed weights and biases */
    static CNNLayer::Ptr createDWConvolutionForScale(const std::string& layerName, size_t channels, float* weights,
                                                     float* biases);

    /**
     * Verifies if layer produces data to layers which marked as float
     */
    static bool layerProducesFloat(const CNNLayer::Ptr layer);

    /**
     * Returns tails from I8 to FP32 until convolution - it is the most performed approach because
     * convolution can convert to FP32 for free, while adding one more scale will decrease performance
     */
    static void returnTailToFP32(const CNNLayer::Ptr layer);

    /**
     * Verifies whether layer can be potentially int8
     * @return true if layer does not have improper activation for fusion
     */
    static bool canLayerBeI8(const CNNLayer::Ptr& layer);

    /**
     * Verifies if next layer has type which potentially can be fused with convolution
     * and if activation is supported for int8
     * @return true if layer does not have improper activation for fusion
     */
    static bool isNextFusionAllowed(const CNNLayer::Ptr& layer);

public:
    /**
     * Returns true for a "relu-like" clamp layer i.e. a clamp with minimum = 0
     */
    static bool isReLULikeClamp(CNNLayer::Ptr layer);
};

typedef std::shared_ptr<CNNNetworkInt8Normalizer> CNNNetworkNormalizerPtr;

}  // namespace details
}  // namespace InferenceEngine
