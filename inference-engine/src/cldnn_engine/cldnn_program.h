// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <map>
#include <set>
#include <memory>
#include <string>
#include <utility>
#include <algorithm>

#include <cpp/ie_cnn_network.h>
#include <cpp_interfaces/exception2status.hpp>
#include <ie_blob.h>
#include <ie_plugin.hpp>

#include "debug_options.h"
#include "cldnn_custom_layer.h"
#include "cldnn_config.h"

#include <api/engine.hpp>
#include <api/memory.hpp>
#include <api/topology.hpp>
#include <api/primitive.hpp>
#include <api/softmax.hpp>
#include <api/resample.hpp>
#include <api/pooling.hpp>
#include <api/eltwise.hpp>
#include <api/concatenation.hpp>
#include <api/detection_output.hpp>

namespace CLDNNPlugin {
template<typename LayerTypePtr>
LayerTypePtr tryAs(const InferenceEngine::CNNLayerPtr& in_ptr) {
    return dynamic_cast<LayerTypePtr>(in_ptr.get());
}

template<typename LayerTypePtr>
LayerTypePtr as(const InferenceEngine::CNNLayerPtr& in_ptr) {
    auto result_ptr = dynamic_cast<LayerTypePtr> (in_ptr.get());
    if (nullptr == result_ptr) {
        THROW_IE_EXCEPTION << "CNNLayerPtr is not suitable for casting to requested layer type";
    }
    return result_ptr;
}

inline std::string layer_type_lower(const InferenceEngine::CNNLayer* layer) {
    std::string layerType = layer->type;
    std::transform(layerType.begin(), layerType.end(), layerType.begin(),
        [](unsigned char c) -> unsigned char { return std::tolower(c); });
    return layerType;
}

inline std::string layer_type_name_ID(const InferenceEngine::CNNLayer* layer) {
    return layer_type_lower(layer) + ":" + layer->name;
}

inline std::string layer_type_lower(InferenceEngine::CNNLayerPtr layer) {
    return layer_type_lower(layer.get());
}

inline std::string layer_type_name_ID(InferenceEngine::CNNLayerPtr layer) {
    return layer_type_name_ID(layer.get());
}

struct PerfCounter {
    InferenceEngine::InferenceEngineProfileInfo::LayerStatus status;
    bool isCPU;
    uint64_t realTime_uSec;
    uint64_t cpu_uSec;
    uint32_t num;
    std::string layerType;
    std::string parentPrimitive;

public:
    PerfCounter() : realTime_uSec(0), cpu_uSec(0), num(0),
                    status(InferenceEngine::InferenceEngineProfileInfo::NOT_RUN), isCPU(false) {}

    long long realTime_avg() const { return (num == 0) ? 0 : realTime_uSec / num; }
    long long cpu_avg() const { return (num == 0) ? 0 : cpu_uSec / num; }
};

class Program {
public:
    Program(InferenceEngine::ICNNNetwork &network, std::shared_ptr<const cldnn::engine> engine, const Config& config);
    std::shared_ptr<cldnn::program> getCompiledProgram(int program_id = 0);

    std::map<std::string, cldnn::primitive_id> primitiveIDs;
    std::map<cldnn::primitive_id, std::vector<std::string>> primitivesToIRLayersMap;
    std::map<std::string, std::vector<cldnn::primitive_id>> prevPrimitiveIDs;
    std::map<cldnn::primitive_id, std::pair<std::string, PerfCounter>> perfMap;

    std::vector<cldnn::primitive_id> profilingIDs;

    std::map<std::string, InferenceEngine::SizeVector> outputDims;
    std::map<std::string, cldnn::layout> inputLayouts;
    std::map<const char *, cldnn::primitive_id> blobMemCache;

    int m_max_batch;
    int m_curBatch;

    InferenceEngine::OutputsDataMap p_currentOutputs;

    std::vector<cldnn::primitive_id> GetPrevLayersPrimitives(const InferenceEngine::CNNLayerPtr layer) const;
    const std::map<std::string, cldnn::layout>& getInputLayouts() const { return inputLayouts; }
    int GetMaxBatchSizeForSingleProgram();

    void AddPrimitiveToProfiler(cldnn::primitive_id id, const InferenceEngine::CNNLayerPtr &layer,
                                cldnn::primitive_id customOutputId = "");

    void AddInnerPrimitiveToProfiler(cldnn::primitive_id id, cldnn::primitive_id parentId,
                                     const InferenceEngine::CNNLayerPtr &layer);

    // internal types
    enum LayerType {
        Convolution,
        DeformableConvolution,
        ReLU,
        ReLU6,
        Sigmoid,
        TanH,
        ELU,
        Activation,
        Exp,
        Asin,
        Atan,
        Acos,
        Abs,
        Asinh,
        Acosh,
        Atanh,
        Not,
        LRN,
        Pooling,
        FullyConnected,
        SoftMax,
        Power,
        Split,
        VariadicSplit,
        Concatenate,
        Eltwise,
        SimplerNMS,
        ROIPooling,
        Crop,
        Deconvolution,
        PriorBox,
        DetectionOutput,
        Normalize,
        Reshape,
        Transpose,
        Permute,
        Flatten,
        BatchNormalization,
        PReLU,
        ScaleShift,
        Proposal,
        PSROIPooling,
        Clamp,
        Copy,
        Resample,
        Interp,
        RegionYolo,
        ReorgYolo,
        ConstantBlob,
        ArgMax,
        ArgMin,
        MVN,
        Unpooling,
        Tile,
        Pad,
        LSTMCell,
        RNN,
        Gather,
        DepthToSpace,
        SpaceToDepth,
        ShuffleChannels,
        StridedSlice,
        Broadcast,
        ReverseSequence,
        BinaryConvolution,
        Quantize,
        Squeeze,
        Unsqueeze,
        Reduce,
        TopK,
        Floor,
        Ceil,
        Erf,
        HardSigmoid,
        Log,
        Neg,
        Reciprocal,
        Selu,
        Sign,
        SoftPlus,
        SoftSign,
        Swish,
        Gelu,
        Sin,
        Sinh,
        Cos,
        Cosh,
        Tan,
        Gemm,
        OneHot,
        Convert,
        ConvertLike,
        GatherTree,
        ExperimentalDetectronROIFeatureExtractor,
        NonMaxSuppression,
        Select,
        GRN,
        CTCGreedyDecoder,
        PriorBoxClustered,
        NO_TYPE
    };
    using GenericBlobMap = std::map<cldnn::primitive_id, cldnn::primitive_id>;

    static LayerType LayerTypeFromStr(const std::string& str);

private:
    std::vector<std::shared_ptr<cldnn::program>> m_programs;
    std::shared_ptr<const cldnn::engine> m_engine;
    Config m_config;

    std::shared_ptr<cldnn::program> BuildProgram(InferenceEngine::ICNNNetwork &network);

    void InitProfileInfo(const std::string& layerName,
                         const std::string& layerType,
                         bool isCPU = false,
                         InferenceEngine::InferenceEngineProfileInfo::LayerStatus status
                         = InferenceEngine::InferenceEngineProfileInfo::EXECUTED,
                         std::string parentId = "");

    static const cldnn::primitive_id m_preProcessTag;
    static const cldnn::primitive_id m_weightsTag;
    static const cldnn::primitive_id m_biasesTag;
    static const cldnn::primitive_id m_meanValuesTag;
    static const cldnn::primitive_id m_postProcessTag;
    static const cldnn::primitive_id m_scalesTag;
    static const cldnn::primitive_id m_workaroundTag;
    static const cldnn::primitive_id m_preCustomLayerTag;
    static const cldnn::primitive_id m_postCustomLayerTag;


    enum WeightRearrangeType {
        BroadcastFeatures,
        FlipDeconvDims,
        NO_REARRANGE
    };

    cldnn::format m_defaultFormat;
    void InitFormat(InferenceEngine::ICNNNetwork &network);

    static cldnn::resample_type ResampleTypeFromString(const std::string &str);

    void Load(InferenceEngine::ICNNNetwork &network);
    static cldnn::pooling_mode PoolingModeFromIEPooling(InferenceEngine::PoolingLayer::PoolType pt, bool excludePadding = false);
    static cldnn::eltwise_mode EltwiseModeFromIEEltwise(InferenceEngine::EltwiseLayer::eOperation op);
    static cldnn::prior_box_code_type PriorBoxCodeFromString(const std::string& str);
    static cldnn::softmax::dimension_t SoftmaxDimensionFromIEAxis(const InferenceEngine::SoftMaxLayer* softmaxLayer);
    cldnn::primitive_id CreatePrimitiveFromBlob(cldnn::topology& topology,
                                                cldnn::primitive_id primID,
                                                const InferenceEngine::Blob::Ptr pBlob,
                                                const cldnn::layout& blobLayout,
                                                size_t blobByteOffset = 0,
                                                WeightRearrangeType rearrange = NO_REARRANGE);
    void CreateWeightAndBiasPrimitives(cldnn::topology& topology,
                                       const InferenceEngine::CNNLayerPtr& layer,
                                       std::vector<cldnn::primitive_id>& weightsPrimID,
                                       std::vector<cldnn::primitive_id>& biasesPrimID);
    void CreateBinaryWeightAndBiasPrimitives(cldnn::topology& topology,
                                             const InferenceEngine::CNNLayerPtr& layer,
                                             std::vector<cldnn::primitive_id>& weightsPrimID,
                                             std::vector<cldnn::primitive_id>& biasesPrimID);
    void CreateScaleWeightsAndBiasesFromBN(cldnn::topology& topology,
                                           const InferenceEngine::BatchNormalizationLayer* bnLayer,
                                           cldnn::primitive_id& weightsPrimID,
                                           cldnn::primitive_id& biasesPrimID);
    void AddPreProcessPrimitive(InferenceEngine::InputInfo::Ptr inputInfo);
    void AddInputPrimitive(cldnn::topology& topology,
                           InferenceEngine::InputInfo::Ptr inputInfo, InferenceEngine::Precision inputPrecision, const std::string inputName);
    void AddOutputPrimitive(cldnn::topology& topology,
                            std::string outputName, const InferenceEngine::DataPtr outputData,
                            InferenceEngine::Precision outputPrecision = InferenceEngine::Precision::UNSPECIFIED);
    void CreateSingleLayerPrimitive(cldnn::topology& topology, InferenceEngine::CNNLayerPtr& layer);
    bool IsValidSplitConvMerge(const InferenceEngine::SplitLayer* splitLayer) const;
    bool CanProcessDynBatch(InferenceEngine::ICNNNetwork &network) const;
    static std::vector<InferenceEngine::CNNLayerPtr> GetNextLayers(const InferenceEngine::DataPtr data);
    static std::vector<InferenceEngine::CNNLayerPtr> GetNextLayers(const InferenceEngine::CNNLayerPtr layer);
    static InferenceEngine::CNNLayerPtr GetNextSingleLayer(const InferenceEngine::DataPtr data);
    static InferenceEngine::CNNLayerPtr GetNextSingleLayer(const InferenceEngine::CNNLayerPtr layer);
    void AddSingleValuePrimitive(cldnn::topology& topology, cldnn::primitive_id valPrimID, cldnn::data_types dataType, float value);

    GenericBlobMap CreateGenericLayerBlobPrimitives(cldnn::topology& topology, const InferenceEngine::GenericLayer* layer);
    static void ValidateGenericLayerBlobs(const InferenceEngine::GenericLayer* layer, const std::vector<std::string>& blobNames);
    static bool HasParam(const std::map<std::string, std::string>& layerParams, std::string paramName) {
        auto p = layerParams.find(paramName);
        return p != layerParams.end();
    }

    void changeInputBatch(int batch);

    // Layer Primitive Creators
    void CreatePReLUPrimitive(cldnn::topology& topology, InferenceEngine::CNNLayerPtr &layer);
    void CreateBatchNormalizationPrimitive(cldnn::topology& topology, InferenceEngine::CNNLayerPtr & layer);
    void CreateFlattenPrimitive(cldnn::topology& topology, InferenceEngine::CNNLayerPtr &layer);
    void CreatePermutePrimitive(cldnn::topology& topology, InferenceEngine::CNNLayerPtr &layer);
    void CreateReshapePrimitive(cldnn::topology& topology, InferenceEngine::CNNLayerPtr &layer);
    void CreateNormalizePrimitive(cldnn::topology& topology, InferenceEngine::CNNLayerPtr &layer);
    void CreateDetectionOutputPrimitive(cldnn::topology& topology, InferenceEngine::CNNLayerPtr &layer);
    void CreatePriorBoxPrimitive(cldnn::topology& topology, InferenceEngine::CNNLayerPtr &layer);
    void CreateDeconvolutionPrimitive(cldnn::topology& topology, InferenceEngine::CNNLayerPtr &layer);
    void CreateCropPrimitive(cldnn::topology& topology, InferenceEngine::CNNLayerPtr &layer);
    void CreateROIPoolingPrimitive(cldnn::topology& topology, InferenceEngine::CNNLayerPtr &layer);
    void CreateSimplerNMSPrimitive(cldnn::topology& topology, InferenceEngine::CNNLayerPtr &layer);
    void CreateEltwisePrimitive(cldnn::topology& topology, InferenceEngine::CNNLayerPtr &layer);
    void CreateConcatenatePrimitive(cldnn::topology& topology, InferenceEngine::CNNLayerPtr &layer);
    void CreateSplitPrimitive(cldnn::topology& topology, InferenceEngine::CNNLayerPtr &layer);
    void CreateFusedSplitConvMergePrimitive(cldnn::topology& topology, InferenceEngine::CNNLayerPtr &layer, bool useGroups = true);
    void CreatePowerPrimitive(cldnn::topology& topology, InferenceEngine::CNNLayerPtr &layer);
    void CreateSoftMaxPrimitive(cldnn::topology& topology, InferenceEngine::CNNLayerPtr &layer);
    void CreateFullyConnectedPrimitive(cldnn::topology& topology, InferenceEngine::CNNLayerPtr &layer);
    void CreatePoolingPrimitive(cldnn::topology& topology, InferenceEngine::CNNLayerPtr &layer);
    void CreateLRNPrimitive(cldnn::topology& topology, InferenceEngine::CNNLayerPtr &layer);
    void CreateActivationPrimitive(cldnn::topology& topology, InferenceEngine::CNNLayerPtr &layer, const LayerType type);
    void CreateConvolutionPrimitive(cldnn::topology& topology, InferenceEngine::CNNLayerPtr &layer);
    void CreateDeformableConvolutionPrimitive(cldnn::topology& topology, InferenceEngine::CNNLayerPtr &layer);
    void CreateScaleShiftPrimitive(cldnn::topology& topology, InferenceEngine::CNNLayerPtr &layer);
    void CreateProposalPrimitive(cldnn::topology& topology, InferenceEngine::CNNLayerPtr &layer);
    void CreatePSROIPoolingPrimitive(cldnn::topology& topology, InferenceEngine::CNNLayerPtr &layer);
    void CreateCopyPrimitive(cldnn::topology& topology, InferenceEngine::CNNLayerPtr &layer);
    void CreateResamplePrimitive(cldnn::topology& topology, InferenceEngine::CNNLayerPtr &layer);
    void CreateInterpPrimitive(cldnn::topology& topology, InferenceEngine::CNNLayerPtr &layer);
    void CreateYOLO2RegionPrimitive(cldnn::topology& topology, InferenceEngine::CNNLayerPtr &layer);
    void CreateYOLO2ReorgPrimitive(cldnn::topology& topology, InferenceEngine::CNNLayerPtr &layer);
    void CreateArgMaxMinPrimitive(cldnn::topology& topology, InferenceEngine::CNNLayerPtr &layer, const LayerType type);
    void CreateTopKPrimitive(cldnn::topology& topology, InferenceEngine::CNNLayerPtr &layer);
    void CreateMaxUnpoolingPrimitive(cldnn::topology& topology, InferenceEngine::CNNLayerPtr &layer);
    void CreateMVNPrimitive(cldnn::topology& topology, InferenceEngine::CNNLayerPtr &layer);
    void CreateTilePrimitive(cldnn::topology& topology, InferenceEngine::CNNLayerPtr &layer);
    void CreatePadPrimitive(cldnn::topology& topology, InferenceEngine::CNNLayerPtr &layer);
    void CreateRegularLSTM(cldnn::topology& topology, InferenceEngine::CNNLayerPtr &layer);
    void CreateDynamicLSTM(cldnn::topology& topology, InferenceEngine::CNNLayerPtr &layer);
    void CreateRNNPrimitive(cldnn::topology& topology, InferenceEngine::CNNLayerPtr &layer);
    void CreateLSTMCellPrimitive(cldnn::topology& topology, InferenceEngine::CNNLayerPtr &layer);
    void AddConstantBlobInput(cldnn::topology& topology, InferenceEngine::CNNLayerPtr &layer);
    void CreateCustomLayerPrimitive(cldnn::topology& topology, InferenceEngine::CNNLayerPtr &layer, CLDNNCustomLayerPtr customLayer);
    void CreateGatherPrimitive(cldnn::topology& topology, InferenceEngine::CNNLayerPtr &layer);
    void CreateDepthToSpacePrimitive(cldnn::topology& topology, InferenceEngine::CNNLayerPtr &layer);
    void CreateSpaceToDepthPrimitive(cldnn::topology& topology, InferenceEngine::CNNLayerPtr &layer);
    void CreateShuffleChannelsPrimitive(cldnn::topology& topology, InferenceEngine::CNNLayerPtr &layer);
    void CreateStridedSlicePrimitive(cldnn::topology& topology, InferenceEngine::CNNLayerPtr &layer);
    void CreateBroadcastPrimitive(cldnn::topology &topology, InferenceEngine::CNNLayerPtr &layer);
    void CreateReverseSequencePrimitive(cldnn::topology& topology, InferenceEngine::CNNLayerPtr &layer);
    void CreateBinaryConvolutionPrimitive(cldnn::topology& topology, InferenceEngine::CNNLayerPtr &layer);
    void CreateQuantizePrimitive(cldnn::topology& topology, InferenceEngine::CNNLayerPtr &layer);
    void CreateGemmPrimitive(cldnn::topology& topology, InferenceEngine::CNNLayerPtr &layer);
    void CreateReducePrimitive(cldnn::topology& topology, InferenceEngine::CNNLayerPtr &layer);
    void CreateOneHotPrimitive(cldnn::topology& topology, InferenceEngine::CNNLayerPtr &layer);
    void CreateGatherTreePrimitive(cldnn::topology& topology, InferenceEngine::CNNLayerPtr &layer);
    void CreateConvertPrimitive(cldnn::topology& topology, InferenceEngine::CNNLayerPtr &layer);
    void CreateConvertLikePrimitive(cldnn::topology& topology, InferenceEngine::CNNLayerPtr &layer);
    void CreatePyramidRoIAlignPrimitive(cldnn::topology& topology, InferenceEngine::CNNLayerPtr &layer);
    void CreateNonMaxSuppressionPrimitive(cldnn::topology& topology, InferenceEngine::CNNLayerPtr &layer);
    void CreateSelectPrimitive(cldnn::topology& topology, InferenceEngine::CNNLayerPtr& layer);
    void CreateGRNPrimitive(cldnn::topology& topology, InferenceEngine::CNNLayerPtr& layer);
    void CreateCTCGreedyDecoderPrimitive(cldnn::topology& topology, InferenceEngine::CNNLayerPtr& layer);
    void CreatePriorBoxClusteredPrimitive(cldnn::topology& topology, InferenceEngine::CNNLayerPtr& layer);
};

}  // namespace CLDNNPlugin
