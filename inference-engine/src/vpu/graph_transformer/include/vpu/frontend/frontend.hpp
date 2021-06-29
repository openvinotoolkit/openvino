// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <tuple>
#include <set>

#include <ie_icore.hpp>
#include <caseless.hpp>
#include <cpp/ie_cnn_network.h>

#include <vpu/stage_builder.hpp>
#include <vpu/frontend/ie_parsed_network.hpp>
#include <vpu/model/model.hpp>
#include <vpu/utils/enums.hpp>
#include <vpu/utils/func_ref.hpp>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/opsets/opset4.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <ngraph/opsets/opset6.hpp>
#include <ngraph/util.hpp>
#include "transformations/utils/utils.hpp"
namespace vpu {
enum EltwiseOperation {
    Sub = 0,
    Sum,          
    Prod,         
    Max,          
    Div,          
    Min,          
    Squared_diff, 
    Equal,        
    Not_equal,    
    Greater,      
    Greater_equal,
    Less,         
    Less_equal,   
    Logical_NOT,  
    Logical_AND,  
    Logical_OR,   
    Logical_XOR,  
    Pow,          
    Floor_mod,    
};
enum PoolNDMethod   { PoolND_max = 1, PoolND_avg = 2 };

enum PoolNDRounding { PoolND_floor = 3, PoolND_ceil  = 4 };

enum PoolMethod   { Pool_max = 1, Pool_avg = 2 };
struct PoolingParams {
    std::vector<size_t> kernel;
    std::vector<size_t> strides;
    std::vector<size_t> padsBegin;
    std::vector<size_t> padsEnd;
    std::string autoPad;
    bool excludePad;
    PoolMethod poolMethod;

};

using OutNode = ngraph::Output<ngraph::Node>;
using NodePtr = std::shared_ptr<ngraph::Node>;
namespace ie = InferenceEngine;

class FrontEnd final {
public:
    using Ptr = std::shared_ptr<FrontEnd>;

    explicit FrontEnd(StageBuilder::Ptr stageBuilder, const ie::ICore* core);

    ModelPtr buildInitialModel(const ie::CNNNetwork& network);

    std::set<std::string> checkSupportedLayers(const ie::CNNNetwork& network);

    const ngraph::NodeVector& origNodes() const {
        return _ieParsedNetwork.orderedOps;
    } 

//
// Passes
//

private:
    ModelPtr runCommonPasses(const ie::CNNNetwork& network);

    using SupportedNodeCallback = std::function<void(const NodePtr&)>;
    using UnsupportedNodeCallback = std::function<void(const Model&, const NodePtr&, const DataVector&, const DataVector&, const std::string&)>;
    ModelPtr runCommonPasses(ie::CNNNetwork network, const UnsupportedNodeCallback& unsupportedLayer,
                             const SupportedNodeCallback& supportedLayer = nullptr);

    //
    // Update IE Network
    //

    void unrollLoops(
            ie::CNNNetwork& network);

    void detectNetworkBatch(
            ie::CNNNetwork& network,
            const Model& model);

    void removeConstLayers(
            ie::CNNNetwork& network);

    //
    // Process internal VPU Model
    //

    void parseInputAndOutputData(const Model& model);

    void addDataTypeConvertStages(const Model& model);

    void addPreProcessStages(const Model& model);

//
// IR Parsers
//

public:
    //
    // Layers that might be both SW and HW
    //

    void parseConvolution(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const; // reworked
    void parseGroupConvolution(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const; // reworked
    void parseAvgPooling(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;
    void parseMaxPooling(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;
    void parseFullyConnected(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const; // rework

    //
    // SW only layers
    //

    void parseReLU(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;  // ok
    void parseSoftMax(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;  // lgtm
    void parseGRN(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;  // lgtm
    void parseMVN(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;  // lgtm
    void parseNorm(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;  // ok
    void parsePower(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const; // need to validate
    void parseSqrt(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const; // --/--/
    void parseScale(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const; // convert_mul_add_to_scaleshift_or_power.cpp
    void parsePermute(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;   // ok
    void parseDetectionOutput(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const; // need to validate keep_top_k??
    // void parseEltwise(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;  // need to rework logic
    void parseSubtract(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;
    void parseAdd(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;
    void parseMultiply(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;
    void parseMaximum(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;
    void parseDivide(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;
    void parseMinimum(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;
    void parseSquaredDifference(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;
    void parseEqual(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;
    void parseNotEqual(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;
    void parseGreater(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;
    void parseGreaterEqual(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;
    void parseLess(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;
    void parseLessEqual(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;
    void parseLogicalNot(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;
    void parseLogicalAnd(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;
    void parseLogicalOr(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;
    void parseLogicalXor(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;

    void parseSigmoid(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;  // lgtm
    void parseTanH(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;  //lgtm
    void parsePReLU(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;  // rework, share_weights_
    // void parseBatchNorm(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;  // ok?
    void parseDeconvolution(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;  // ok
    void parseCopy(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;  // lgtm
    void parseELU(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const; // lgtm
    void parseCrop(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const; // ngraph crop representation stridedslice
    void parseTile(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;  // not sure
    void parseNormalize(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const; // ok
    void parseRegionYolo(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;  // lgtm
    void parseReorgYolo(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;  // not sure
    void parseBias(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;  // not sure
    void parseCTCDecoder(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;  // lgtm
    void parseInterp(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;  // need to add fnct ngraph::interp::mode->vpu::mode
    void parseClamp(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;  // lgtm
    void parseProposal(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;  //  not sure
    void parseROIPooling(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;  // lgtm
    void parsePSROIPooling(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;  // lgtm
    // void parseMTCNN(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;  // need to remove?
    void parsePad(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;  //  need to investigate more
    void parseResample(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;  // is it interpolate? CVS-31987
    void parseInterpolate(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;  // need to rework little bit
    void parseRNN(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;  // rework, share weights
    void parseGEMM(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;  // need to investigate
    void parseLog(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;  // not sure
    void parseExp(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;  // not sure
    void parseReverseSequence(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;  // lgtm
    void parseGather(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;  // not sure
    void parseReduceAnd(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;
    void parseReduceMin(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;
    void parseReduceMax(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;
    void parseReduceSum(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;
    void parseReduceMean(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;
    void parseFloor(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;  // lgtm
    void parseTopK(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;  // lgtm
    void parseSelect(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;  // lgtm
    void parseExpDetectionOutput(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;  // lgtm
    void parseROIFeatureExtractor(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;  // ok
    void parseConvert(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;  // lgtm
    void parseErf(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;  // lgtm
    void parseOneHot(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;  // ok
    void parseExpPriorGridGenerator(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const; // lgtm
    void parseExpGenerateProposals(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const; // lgtm
    void parseScatterUpdate(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;  // lgtm
    void parseScatterElementsUpdate(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const; // lgtm
    void parseExpTopKROIs(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;  // lgtm
    void parseNonZero(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;  // lgtm
    void parseROIAlign(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;  // lgtm
    void parseOutShapeOfReshape(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;  // lgtm
    void parseBroadcast(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;  // lgtm
    void parseStaticShapeNMS(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;  // need to rework layer??
    void parseMish(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;  // lgtm
    void parseGelu(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;  // lgtm
    void parseSoftPlus(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;  // lgtm
    void parseSwish(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;  //  ok
    // void parseActivation(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;  // need to investigate
    // void parseLogicalNot(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;  // need to investigate
    void parseGatherND(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;  // lgtm
    void parseHSwish(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;  // lgtm
    void parseCeiling(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;  // lgtm
    void parseRound(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;  // lgtm
    void parseCTCGreedyDecoderSeqLen(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;  //lgtm

    //
    // Special layers
    //

    void parsePriorBox(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;  // lgtm
    void parsePriorBoxClustered(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;  // lgtm
    void parseReshape(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;  // lgtm
    void parseConcat(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;  // lgtm
    void parseSplit(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;  // lgtm
    void parseStridedSlice(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const;  // lgtm
    void parseDSR(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs);  // lgtm
    void parseGatherElements(const Model &model, const NodePtr& node, const DataVector &inputs, const DataVector &outputs) const;  // lgtm

    //
    // Parser with data sharing
    //

    // void parseCustom(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs); // ???
    void parseLSTMCell(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs);  // ok
    // void parseTensorIterator(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs);  // ???

    //
    // Utility
    //
    
    // static CustomLayer::Ptr getSuitableCustomLayer(const std::vector<CustomLayer::Ptr>& customLayers, const NodePtr&cnnLayer);
    void parsePoolingImpl(const Model& model, const NodePtr& node,const DataVector & inputs, const DataVector& outputs, const PoolingParams & params) const;
    void parseEltwiseImpl(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs, const EltwiseOperation eltwiseOperation) const;
    void parseReduceImpl(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs, vpu::StageType reduceType, bool keepDims) const;  // not sure
    static ie::CNNNetwork convertNetwork(ie::CNNNetwork& network);

private:
    Data getVpuData(const OutNode& ieData) const;
    void bindData(const Data& data, const OutNode& nodeOutput, NodePtr origNode);

    void getInputAndOutputData(
            const Model& model,
            const NodePtr& node,
            DataVector& inputs,
            DataVector& outputs);

    static ie::Blob::Ptr shareWeights(const NodePtr& constLayer);
    std::tuple<Data, Data> getWeightsAndBiases(const Model& model, const std::string nodeName, const NodePtr& weightsNode, const NodePtr& biasesNode) const;

    void defaultOnUnsupportedLayerCallback(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs,
                                           const std::string& extraMessage);

    void parseLayer(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs);
    void parseLayer(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs,
                    const UnsupportedNodeCallback& onUnsupported, const SupportedNodeCallback& onSupported = nullptr);

    void processTrivialCases(const Model& model);

private:
    StageBuilder::Ptr _stageBuilder;
    const ie::ICore* _core = nullptr;

    IeParsedNetwork _ieParsedNetwork;
    std::unordered_set<ie::DataPtr> _unbatchedOutputs;
    // ie::details::caseless_map<std::string, std::vector<CustomLayer::Ptr>> _customLayers;

#define LAYER_PARSER(functor_name)                                                                                \
    [this](const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) \
        { functor_name(model, node, inputs, outputs); }

    using LayerParser = std::function<void(const Model&, const NodePtr&, const DataVector&, const DataVector&)>;
    const ie::details::caseless_map<std::string, LayerParser> parsers;

    std::map<OutNode, Data> _ieToVpuMap;
    ie::details::caseless_map<std::string, Data> _kernelNodes;
    std::unordered_map<ie::Blob::Ptr, Data> _lstmWeights;
    std::unordered_map<ie::Blob::Ptr, Data> _lstmBiases;
};

}  // namespace vpu
