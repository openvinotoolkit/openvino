// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <map>
#include <set>
#include <memory>
#include <string>
#include <utility>
#include "ie_blob.h"
#include "ie_plugin.hpp"
#include "cpp/ie_cnn_network.h"
#include "debug_options.h"
#include "inference_engine.hpp"
#include <CPP/network.hpp>
#include <CPP/memory.hpp>
#include <CPP/primitive.hpp>
#include <CPP/topology.hpp>
#include <CPP/pooling.hpp>
#include <CPP/eltwise.hpp>
#include <CPP/concatenation.hpp>
#include <CPP/detection_output.hpp>
#include <CPP/softmax.hpp>
#include <cpp_interfaces/impl/ie_executable_network_thread_safe_default.hpp>
#include <CPP/upsampling.hpp>
#include "cldnn_custom_layer.h"

namespace CLDNNPlugin {

struct PerfCounter {
    InferenceEngine::InferenceEngineProfileInfo::LayerStatus status;
    bool isCPU;
    uint64_t realTime_uSec;
    uint64_t cpu_uSec;
    uint32_t num;
    std::string layerType;

public:
    PerfCounter() : realTime_uSec(0), cpu_uSec(0), num(0),
        status(InferenceEngine::InferenceEngineProfileInfo::NOT_RUN), isCPU(false) {}

    long long realTime_avg() const { return (num == 0) ? 0 : realTime_uSec / num; }
    long long cpu_avg() const { return (num == 0) ? 0 : cpu_uSec / num; }
};

struct InferenceEnv {
    std::shared_ptr<const cldnn::engine> engine;
    std::shared_ptr<cldnn::network> network;
    std::map<std::string, cldnn::primitive_id> primitiveIDs;
    std::map<std::string, std::vector<cldnn::primitive_id>> prevPrimitiveIDs;

    std::map<cldnn::primitive_id, std::pair<std::string, PerfCounter>> perfMap;
    std::vector<cldnn::primitive_id> profilingIDs;

    DebugOptions debugOptions;

    std::map<std::string, InferenceEngine::SizeVector> outputDims;
    std::map<std::string, cldnn::layout> inputLayouts;

    std::vector<std::shared_ptr<cldnn::network>> batchNetworks;
    int m_max_batch;
    int m_bv_sz;
};

class CLDNNGraph : public InferenceEngine::ExecutableNetworkThreadSafeDefault {
public:
    typedef std::shared_ptr<CLDNNGraph> Ptr;
    struct Config {
        Config() : useProfiling(false), dumpCustomKernels(false), exclusiveAsyncRequests(false),
            memory_pool_on(true),
            enableDynamicBatch(false),
            queuePriority(cldnn::priority_mode_types::disabled),
            queueThrottle(cldnn::throttle_mode_types::disabled) {}

        void LoadFromMap(const std::map<std::string, std::string>& configMap);

        bool enableDynamicBatch;
        bool useProfiling;
        bool dumpCustomKernels;
        bool exclusiveAsyncRequests;
        bool memory_pool_on;
        cldnn::priority_mode_types queuePriority;
        cldnn::throttle_mode_types queueThrottle;
        CLDNNCustomLayerMap customLayers;
        cldnn::tuning_config_options tuningConfig;
        std::string graph_dumps_dir;
        std::string sources_dumps_dir;
    };
    explicit CLDNNGraph(InferenceEngine::ICNNNetwork &network, const Config& config = {}, int max_batch = -1);

    InferenceEngine::InferRequestInternal::Ptr
    CreateInferRequestImpl(InferenceEngine::InputsDataMap networkInputs, InferenceEngine::OutputsDataMap networkOutputs) override;

    static bool IsLayerSupported(const std::string &type) {
        return LayerTypeFromStr(type) != NO_TYPE;
    }

protected:
    // graph members
    std::shared_ptr<cldnn::topology> m_topology;
    InferenceEnv m_env;
    Config m_config;

    InferenceEngine::InputsDataMap*  p_currentInputs;
    InferenceEngine::OutputsDataMap* p_currentOutputs;
    int m_curBatch;
    static const cldnn::primitive_id m_preProcessTag;
    static const cldnn::primitive_id m_weightsTag;
    static const cldnn::primitive_id m_biasesTag;
    static const cldnn::primitive_id m_meanValuesTag;
    static const cldnn::primitive_id m_postProcessTag;
    static const cldnn::primitive_id m_scalesTag;
    static const cldnn::primitive_id m_workaroundTag;
    static const cldnn::primitive_id m_preCustomLayerTag;
    static const cldnn::primitive_id m_postCustomLayerTag;

    // internal types
    enum LayerType {
        Convolution,
        ReLU,
        ReLU6,
        Sigmoid,
        TanH,
        ELU,
        Activation,
        Exp,
        Not,
        LRN,
        Pooling,
        FullyConnected,
        SoftMax,
        Power,
        Split,
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
        Permute,
        Flatten,
        BatchNormalization,
        PReLU,
        ScaleShift,
        Proposal,
        PSROIPooling,
        Clamp,
        Copy,
        Upsampling,
        Resample,
        RegionYolo,
        ReorgYolo,
        ConstantBlob,
        ArgMax,
        MVN,
        Unpooling,
        Tile,
        Pad,
        LSTMCell,
        RNN,
        Gather,
        DepthToSpace,
        ShuffleChannels,
        StridedSlice,
        ReverseSequence,
        NO_TYPE
    };

    enum WeightRearrangeType {
        BroadcastFeatures,
        FlipDeconvDims,
        NO_REARRANGE
    };

    cldnn::format m_defaultFormat;
    void InitFormat(InferenceEngine::ICNNNetwork &network);

    static cldnn::data_types DataTypeFromPrecision(InferenceEngine::Precision p);
    static cldnn::format     FormatFromLayout(InferenceEngine::Layout l);
    static cldnn::upsampling_sample_type UpsamplingTypeFromString(const std::string& str);

    void Load(InferenceEngine::ICNNNetwork &network);
    static LayerType LayerTypeFromStr(const std::string& str);
    static cldnn::pooling_mode PoolingModeFromIEPooling(InferenceEngine::PoolingLayer::PoolType pt, bool excludePadding = false);
    static cldnn::eltwise_mode EltwiseModeFromIEEltwise(InferenceEngine::EltwiseLayer::eOperation op);
    static cldnn::concatenation::concatenation_axis ConcatAxisFromIEAxis(unsigned axis);
    static cldnn::prior_box_code_type PriorBoxCodeFromString(const std::string& str);
    static cldnn::softmax::dimension_t SoftmaxDimensionFromIEAxis(const InferenceEngine::SoftMaxLayer* softmaxLayer, bool isPrevFC = false);
    void CreatePrimitiveFromBlob(cldnn::primitive_id primID,
                                 const InferenceEngine::Blob::Ptr pBlob,
                                 cldnn::layout blobLayout,
                                 size_t blobByteOffset = 0,
                                 WeightRearrangeType rearrange = NO_REARRANGE);
    void CreateWeightAndBiasPrimitives(const InferenceEngine::CNNLayerPtr& layer,
                                       std::vector<cldnn::primitive_id>& weightsPrimID,
                                       std::vector<cldnn::primitive_id>& biasesPrimID);
    void CreateScaleWeightsAndBiasesFromBN(const InferenceEngine::BatchNormalizationLayer* bnLayer,
                                           cldnn::primitive_id weightsPrimID,
                                           cldnn::primitive_id biasesPrimID);
    void AddPreProcessPrimitive(InferenceEngine::InputInfo::Ptr inputInfo);
    void AddInputPrimitive(InferenceEngine::InputInfo::Ptr inputInfo, InferenceEngine::Precision inputPrecision);
    void AddOutputPrimitive(std::string outputName, const InferenceEngine::DataPtr outputData,
                            InferenceEngine::Precision outputPrecision = InferenceEngine::Precision::UNSPECIFIED);
    void CreateSingleLayerPrimitive(InferenceEngine::CNNLayerPtr& layer);
    bool IsValidSplitConvMerge(const InferenceEngine::SplitLayer* splitLayer) const;
    bool CanProcessDynBatch(InferenceEngine::ICNNNetwork &network) const;
    static std::vector<InferenceEngine::CNNLayerPtr> GetNextLayers(const InferenceEngine::DataPtr data);
    static std::vector<InferenceEngine::CNNLayerPtr> GetNextLayers(const InferenceEngine::CNNLayerPtr layer);
    static InferenceEngine::CNNLayerPtr GetNextSingleLayer(const InferenceEngine::DataPtr data);
    static InferenceEngine::CNNLayerPtr GetNextSingleLayer(const InferenceEngine::CNNLayerPtr layer);
    std::vector<cldnn::primitive_id> GetPrevLayersPrimitives(const InferenceEngine::CNNLayerPtr layer) const;
    void AddSingleValuePrimitive(cldnn::primitive_id valPrimID, cldnn::data_types dataType, float value);

    void CreateGenericLayerBlobPrimitives(const InferenceEngine::GenericLayer* layer);
    static void ValidateGenericLayerBlobs(const InferenceEngine::GenericLayer* layer, const std::vector<std::string>& blobNames);
    static cldnn::tensor CldnnTensorFromIEDims(const InferenceEngine::SizeVector& dims);
    static bool HasParam(const std::map<std::string, std::string>& layerParams, std::string paramName) {
        auto p = layerParams.find(paramName);
        return p != layerParams.end();
    }

    void InitProfileInfo(const std::string& layerName,
                         const std::string& layerType,
                         bool isCPU = false,
                         InferenceEngine::InferenceEngineProfileInfo::LayerStatus status
                                    = InferenceEngine::InferenceEngineProfileInfo::EXECUTED);
    void changeInputBatch(size_t batch);
    void CompileNetwork();

    // Layer Primitive Creators
    void CreatePReLUPrimitive(InferenceEngine::CNNLayerPtr &layer);
    void CreateBatchNormalizationPrimitive(InferenceEngine::CNNLayerPtr & layer);
    void CreateFlattenPrimitive(InferenceEngine::CNNLayerPtr &layer);
    void CreatePermutePrimitive(InferenceEngine::CNNLayerPtr &layer);
    void CreateReshapePrimitive(InferenceEngine::CNNLayerPtr &layer);
    void CreateNormalizePrimitive(InferenceEngine::CNNLayerPtr &layer);
    void CreateDetectionOutputPrimitive(InferenceEngine::CNNLayerPtr &layer);
    void CreatePriorBoxPrimitive(InferenceEngine::CNNLayerPtr &layer);
    void CreateDeconvolutionPrimitive(InferenceEngine::CNNLayerPtr &layer);
    void CreateCropPrimitive(InferenceEngine::CNNLayerPtr &layer);
    void CreateROIPoolingPrimitive(InferenceEngine::CNNLayerPtr &layer);
    void CreateSimplerNMSPrimitive(InferenceEngine::CNNLayerPtr &layer);
    void CreateEltwisePrimitive(InferenceEngine::CNNLayerPtr &layer);
    void CreateConcatenatePrimitive(InferenceEngine::CNNLayerPtr &layer);
    void CreateSplitPrimitive(InferenceEngine::CNNLayerPtr &layer);
    void CreateFusedSplitConvMergePrimitive(InferenceEngine::CNNLayerPtr &layer);
    void CreatePowerPrimitive(InferenceEngine::CNNLayerPtr &layer);
    void CreateSoftMaxPrimitive(InferenceEngine::CNNLayerPtr &layer);
    void CreateFullyConnectedPrimitive(InferenceEngine::CNNLayerPtr &layer);
    void CreatePoolingPrimitive(InferenceEngine::CNNLayerPtr &layer);
    void CreateLRNPrimitive(InferenceEngine::CNNLayerPtr &layer);
    void CreateActivationPrimitive(InferenceEngine::CNNLayerPtr &layer, const LayerType type);
    void CreateConvolutionPrimitive(InferenceEngine::CNNLayerPtr &layer);
    void CreateScaleShiftPrimitive(InferenceEngine::CNNLayerPtr &layer);
    void CreateProposalPrimitive(InferenceEngine::CNNLayerPtr &layer);
    void CreatePSROIPoolingPrimitive(InferenceEngine::CNNLayerPtr &layer);
    void CreateCopyPrimitive(InferenceEngine::CNNLayerPtr &layer);
    void CreateUpsamplingPrimitive(InferenceEngine::CNNLayerPtr &layer);
    void CreateResamplePrimitive(InferenceEngine::CNNLayerPtr &layer);
    void CreateYOLO2RegionPrimitive(InferenceEngine::CNNLayerPtr &layer);
    void CreateYOLO2ReorgPrimitive(InferenceEngine::CNNLayerPtr &layer);
    void CreateArgMaxPrimitive(InferenceEngine::CNNLayerPtr &layer);
    void CreateMaxUnpoolingPrimitive(InferenceEngine::CNNLayerPtr &layer);
    void CreateMVNPrimitive(InferenceEngine::CNNLayerPtr &layer);
    void CreateTilePrimitive(InferenceEngine::CNNLayerPtr &layer);
    void CreatePadPrimitive(InferenceEngine::CNNLayerPtr &layer);
    void CreateRNNPrimitive(InferenceEngine::CNNLayerPtr &layer);
    void CreateLSTMCellPrimitive(InferenceEngine::CNNLayerPtr &layer);
    void AddConstantBlobInput(InferenceEngine::CNNLayerPtr &layer);
    void CreateCustomLayerPrimitive(InferenceEngine::CNNLayerPtr &layer, CLDNNCustomLayerPtr customLayer);
    void CreateGatherPrimitive(InferenceEngine::CNNLayerPtr &layer);
    void CreateDepthToSpacePrimitive(InferenceEngine::CNNLayerPtr &layer);
    void CreateShuffleChannelsPrimitive(InferenceEngine::CNNLayerPtr &layer);
    void CreateStridedSlicePrimitive(InferenceEngine::CNNLayerPtr &layer);
    void CreateReverseSequencePrimitive(InferenceEngine::CNNLayerPtr &layer);
};

};  // namespace CLDNNPlugin
