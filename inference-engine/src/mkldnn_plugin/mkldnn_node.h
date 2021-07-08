// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_api.h>
#include <memory>
#include <vector>
#include <string>
#include <cassert>
#include <algorithm>
#include <caseless.hpp>
#include "mkldnn_dims.h"
#include "mkldnn_memory.h"
#include "mkldnn_edge.h"
#include "mkldnn_descriptor.h"
#include "mkldnn_selective_build.h"
#include "mkldnn/iml_type_mapper.h"
#include "mkldnn_extension_mngr.h"
#include "mkldnn_primitive.h"
#include "mkldnn_weights_cache.hpp"
#include "mkldnn.hpp"
#include <openvino/itt.hpp>
#include "utils/ngraph_utils.hpp"
#include <ngraph/ops.hpp>
#include <ngraph/node.hpp>
#include <ie_precision.hpp>
#include <nodes/common/tensor_desc_creator.h>
#include "cpu_types.h"
#include "cpu_shape.h"
#include "cpu_memory_desc.h"

namespace MKLDNNPlugin {

using MKLDNNNodePtr = std::shared_ptr<MKLDNNNode>;
using MKLDNNNodeWeakPtr = std::weak_ptr<MKLDNNNode>;

Type TypeFromName(const std::string type);

static std::string NameFromType(Type type) {
    switch (type) {
        case Generic:
            return "Generic";
        case Reorder:
            return "Reorder";
        case Input:
            return "Input";
        case Output:
            return "Output";
        case Convolution:
            return "Convolution";
        case Deconvolution:
            return "Deconvolution";
        case Lrn:
            return "Lrn";
        case Pooling:
            return "Pooling";
        case FullyConnected:
            return "FullyConnected";
        case MatMul:
            return "MatMul";
        case Softmax:
            return "Softmax";
        case Split:
            return "Split";
        case Concatenation:
            return "Concatenation";
        case StridedSlice:
            return "StridedSlice";
        case Reshape:
            return "Reshape";
        case Tile:
            return "Tile";
        case ROIAlign:
            return "ROIAlign";
        case ROIPooling:
            return "ROIPooling";
        case PSROIPooling:
            return "PSROIPooling";
        case DepthToSpace:
            return "DepthToSpace";
        case BatchToSpace:
            return "BatchToSpace";
        case Pad:
            return "Pad";
        case Transpose:
            return "Transpose";
        case SpaceToDepth:
            return "SpaceToDepth";
        case SpaceToBatch:
            return "SpaceToBatch";
        case MemoryOutput:
            return "MemoryOutput";
        case MemoryInput:
            return "MemoryInput";
        case RNNSeq:
            return "RNNSeq";
        case RNNCell:
            return "RNNCell";
        case Eltwise:
            return "Eltwise";
        case FakeQuantize:
            return "FakeQuantize";
        case BinaryConvolution:
            return "BinaryConvolution";
        case DeformableConvolution:
            return "DeformableConvolution";
        case MVN:
            return "MVN";
        case TensorIterator:
            return "TensorIterator";
        case Convert:
            return "Convert";
        case NormalizeL2:
            return "NormalizeL2";
        case ScatterUpdate:
            return "ScatterUpdate";
        case ScatterElementsUpdate:
            return "ScatterElementsUpdate";
        case ScatterNDUpdate:
            return "ScatterNDUpdate";
        case Interpolate:
            return "Interpolate";
        case Reduce:
            return "Reduce";
        case Broadcast:
            return "Broadcast";
        case EmbeddingSegmentsSum:
            return "EmbeddingSegmentsSum";
        case EmbeddingBagPackedSum:
            return "EmbeddingBagPackedSum";
        case EmbeddingBagOffsetsSum:
            return "EmbeddingBagOffsetsSum";
        case Gather:
            return "Gather";
        case GatherElements:
            return "GatherElements";
        case GatherND:
            return "GatherND";
        case OneHot:
            return "OneHot";
        case RegionYolo:
            return "RegionYolo";
        case Select:
            return "Select";
        case Roll:
            return "Roll";
        case ShuffleChannels:
            return "ShuffleChannels";
        case DFT:
            return "DFT";
        case Math:
            return "Math";
        case CTCLoss:
            return "CTCLoss";
        case Bucketize:
            return "Bucketize";
        case CTCGreedyDecoder:
            return "CTCGreedyDecoder";
        case CTCGreedyDecoderSeqLen:
            return "CTCGreedyDecoderSeqLen";
        case CumSum:
            return "CumSum";
        case DetectionOutput:
            return "DetectionOutput";
        case ExperimentalDetectronDetectionOutput:
            return "ExperimentalDetectronDetectionOutput";
        case LogSoftmax:
            return "LogSoftmax";
        case TopK:
            return "TopK";
        case GatherTree:
            return "GatherTree";
        case GRN:
            return "GRN";
        case Range:
            return "Range";
        case Proposal:
            return "Proposal";
        case ReorgYolo:
            return "ReorgYolo";
        case ReverseSequence:
            return "ReverseSequence";
        case ExperimentalDetectronTopKROIs:
            return "ExperimentalDetectronTopKROIs";
        case ExperimentalDetectronROIFeatureExtractor:
            return "ExperimentalDetectronROIFeatureExtractor";
        case ExperimentalDetectronPriorGridGenerator:
            return "ExperimentalDetectronPriorGridGenerator";
        case ExperimentalDetectronGenerateProposalsSingleImage:
            return "ExperimentalDetectronGenerateProposalsSingleImage";
        case ExtractImagePatches:
            return "ExtractImagePatches";
        case NonMaxSuppression:
            return "NonMaxSuppression";
        default:
            return "Unknown";
    }
}

class PortConfigurator {
public:
    PortConfigurator() = default;

    PortConfigurator(MKLDNNPlugin::TensorDescCreatorTypes tensorDescType, InferenceEngine::Precision prc, const Shape& shape,
                     bool constant = false, int inPlace = -1) :
            tensorDescCreator(getTensorDescCreator(tensorDescType)), prc(prc), shape(shape), constant(constant), inPlace(inPlace) {}

    PortConfigurator(MKLDNNPlugin::TensorDescCreatorTypes tensorDescType, InferenceEngine::Precision prc = InferenceEngine::Precision::UNSPECIFIED,
                     bool constant = false, int inPlace = -1) :
            tensorDescCreator(getTensorDescCreator(tensorDescType)), constant(constant), inPlace(inPlace) {}

    MKLDNNPlugin::TensorDescCreator::CreatorConstPtr tensorDescCreator;
    const InferenceEngine::Precision prc;
    const Shape shape;
    bool constant = false;
    int inPlace = -1;

private:
    static MKLDNNPlugin::TensorDescCreator::CreatorConstPtr getTensorDescCreator(MKLDNNPlugin::TensorDescCreatorTypes tensorDescType) {
        auto& creators = MKLDNNPlugin::TensorDescCreator::getCommonCreators();
        if (creators.find(tensorDescType) == creators.end()) {
            IE_THROW() << "Cannot find tensor descriptor creator";
        }
        return creators.at(tensorDescType);
    }
};

struct PortConfig {
    PortConfig() = default;

    PortConfig(const PortConfig& rhs) {
        this->constant = rhs.constant;
        this->inPlace = rhs.inPlace;
        this->desc = rhs.desc->clone();
    }

    PortConfig& operator=(const PortConfig& rhs) {
        this->constant = rhs.constant;
        this->inPlace = rhs.inPlace;
        this->desc = rhs.desc->clone();
        return *this;
    }

    // TODO [DS]: better to make private and const
    bool constant = false;
    int inPlace = -1;
    std::unique_ptr<MemoryDesc> desc;
};

struct NodeConfig {
    bool dynBatchSupport = false;
    std::vector<PortConfig> inConfs;
    std::vector<PortConfig> outConfs;
};

class NodeDesc {
public:
    NodeDesc(const NodeConfig& conf, impl_desc_type type): config(conf) {
        implementationType = type;
    }

    NodeDesc(const NodeDesc &descInfo) = default;
    NodeDesc(NodeDesc &&descInfo) = default;

    NodeDesc &operator=(const NodeDesc &descInfo) = default;

    const NodeConfig& getConfig() const {
        return config;
    }

    void setConfig(const NodeConfig& config) {
        this->config = config;
    }

    impl_desc_type getImplementationType() const {
        return implementationType;
    }

    void setImplementationType(impl_desc_type type) {
        implementationType = type;
    }

private:
    NodeConfig config;
    impl_desc_type implementationType;
};

class MKLDNNNode {
public:
    template<typename T, int N>
    struct Tag {};

    struct PerfCounters {
        PerfCounters(std::string const& name)
            : execute(openvino::itt::handle(name))
            , getSupportedDescriptors(openvino::itt::handle<Tag<MKLDNNNode, 0>>("MKLDNNNode::getSupportedDescriptors"))
            , initSupportedPrimitiveDescriptors(openvino::itt::handle<Tag<MKLDNNNode, 1>>("MKLDNNNode::initSupportedPrimitiveDescriptors"))
            , filterSupportedPrimitiveDescriptors(openvino::itt::handle<Tag<MKLDNNNode, 2>>("MKLDNNNode::filterSupportedPrimitiveDescriptors"))
            , selectOptimalPrimitiveDescriptor(openvino::itt::handle<Tag<MKLDNNNode, 3>>("MKLDNNNode::selectOptimalPrimitiveDescriptor"))
            , createPrimitive(openvino::itt::handle<Tag<MKLDNNNode, 4>>("MKLDNNNode::createPrimitive"))
            , initOptimalPrimitiveDescriptor(openvino::itt::handle<Tag<MKLDNNNode, 5>>("MKLDNNNode::initOptimalPrimitiveDescriptor"))
        {}

        template<typename NodeType>
        void buildClassCounters(const std::string& type_name) {
            getSupportedDescriptors = openvino::itt::handle<Tag<NodeType, 0>>(type_name + "::getSupportedDescriptors");
            initSupportedPrimitiveDescriptors = openvino::itt::handle<Tag<NodeType, 1>>(type_name + "::initSupportedPrimitiveDescriptors");
            filterSupportedPrimitiveDescriptors = openvino::itt::handle<Tag<NodeType, 2>>(type_name + "::filterSupportedPrimitiveDescriptors");
            selectOptimalPrimitiveDescriptor = openvino::itt::handle<Tag<NodeType, 3>>(type_name + "::selectOptimalPrimitiveDescriptor");
            createPrimitive = openvino::itt::handle<Tag<NodeType, 4>>(type_name + "::createPrimitive");
            initOptimalPrimitiveDescriptor = openvino::itt::handle<Tag<NodeType, 5>>(type_name + "::initOptimalPrimitiveDescriptor");
        }

        openvino::itt::handle_t execute;
        openvino::itt::handle_t getSupportedDescriptors;
        openvino::itt::handle_t initSupportedPrimitiveDescriptors;
        openvino::itt::handle_t filterSupportedPrimitiveDescriptors;
        openvino::itt::handle_t selectOptimalPrimitiveDescriptor;
        openvino::itt::handle_t createPrimitive;
        openvino::itt::handle_t initOptimalPrimitiveDescriptor;
    };

    class NodesFactory;
    static NodesFactory & factory();

    virtual ~MKLDNNNode() = default;

    void addEdge(const MKLDNNEdgeWeakPtr& edge);
    void removeEdge(const MKLDNNEdgeWeakPtr& edge);

    virtual void cleanup();
    void remove();

    const std::vector<MKLDNNEdgeWeakPtr> &getParentEdges() const noexcept {
        return parentEdges;
    }

    const std::vector<MKLDNNEdgeWeakPtr> &getChildEdges() const noexcept {
        return childEdges;
    }

    const MKLDNNEdgePtr getParentEdgeAt(size_t idx) const;
    virtual const MKLDNNEdgePtr getChildEdgeAt(size_t idx) const;

    const std::vector<MKLDNNEdgePtr> getParentEdgesAtPort(size_t idx) const;
    const std::vector<MKLDNNEdgePtr> getChildEdgesAtPort(size_t idx) const;

    bool isDropped() {
        return (isEdgesEmpty(childEdges) && isEdgesEmpty(parentEdges));
    }

    const mkldnn::engine& getEngine() const {
        return engine;
    }

    bool isConstant();

    bool isInplace() const;

    bool isFusedWith(Type type) const;

    void addFusedNode(const MKLDNNNodePtr &fusingNode) {
        fusedWith.push_back(fusingNode);
    }

    virtual void fuseInto(MKLDNNNodePtr& parentNode) {
        // The graph supports fusing only of consecutive nodes and some graph logic requires to know through which input port a node was fused into parent one.
        for (int i = 0; i < getParentEdges().size(); i++) {
            if (getParentEdgesAtPort(i)[0]->getParent().get() == parentNode.get()) {
                setFusingPort(i);
                break;
            }
        }

        auto parentFusedNodes = parentNode->getFusedWith();
        if (getFusingPort() < 0 && !parentFusedNodes.empty()) {
            for (int i = 0; i < getParentEdges().size(); i++) {
                if (getParentEdgesAtPort(i)[0]->getParent().get() == parentFusedNodes[parentFusedNodes.size() - 1].get()) {
                    setFusingPort(i);
                    break;
                }
            }
        }

        if (getFusingPort() == -1) {
            IE_THROW() << "Cannot determine fusing port between nodes: " << parentNode->getName() << " and " << getName();
        }

        parentNode->addFusedNode(getParentEdgesAtPort(getFusingPort())[0]->getChild());
        parentNode->addOriginalLayer(getOriginalLayers());
    }

    void clearFusedWith() {
        fusedWith.clear();
    }

    void mergeWith(const MKLDNNNodePtr &merge) {
        mergedWith.push_back(merge);
    }

    const std::vector <MKLDNNNodePtr> &getMergeWith() {
        return mergedWith;
    }

    const std::vector <MKLDNNNodePtr> &getFusedWith() {
        return fusedWith;
    }

    int getFusingPort() const {
        return fusingPort;
    }

    void setFusingPort(int fusingPort) {
        this->fusingPort = fusingPort;
    }

    const std::string &getName() const {
        return name;
    }

    void addOriginalLayer(const std::string& layerName);

    const std::string &getOriginalLayers() const {
        return originalLayers;
    }

    Type getType() const {
        return type;
    }

    const std::vector<NodeDesc>& getSupportedPrimitiveDescriptors() const {
        return supportedPrimitiveDescriptors;
    }

    inline const NodeDesc* getSelectedPrimitiveDescriptor() const {
        if (selectedPrimitiveDescriptorIndex < 0 ||
            selectedPrimitiveDescriptorIndex >= supportedPrimitiveDescriptors.size())
            return nullptr;
        return &supportedPrimitiveDescriptors[selectedPrimitiveDescriptorIndex];
    }

    inline NodeDesc* getSelectedPrimitiveDescriptor() {
        if (selectedPrimitiveDescriptorIndex < 0 ||
            selectedPrimitiveDescriptorIndex >= supportedPrimitiveDescriptors.size())
            return nullptr;
        return &supportedPrimitiveDescriptors[selectedPrimitiveDescriptorIndex];
    }

    void selectPrimitiveDescriptorByIndex(int index) {
        if (index < 0 || index >= supportedPrimitiveDescriptors.size())
            selectedPrimitiveDescriptorIndex = -1;
        else
            selectedPrimitiveDescriptorIndex = index;
    }

    std::string getPrimitiveDescriptorType();

    PerfCount &PerfCounter() { return perfCounter; }

    virtual void setDynamicBatchLim(int lim);

    void resolveNotAllocatedEdges();
    virtual void execute(mkldnn::stream strm);
    virtual void initSupportedPrimitiveDescriptors();

    /**
     * @brief Filters supportedPrimitiveDescriptors according to the input layouts specified in inputMemoryFormatsFilter
     * and output layouts specified in outputMemoryFormatsFilter
     */
    virtual void filterSupportedPrimitiveDescriptors();

    virtual void createPrimitive() = 0;

    virtual void selectOptimalPrimitiveDescriptor();
    virtual void initOptimalPrimitiveDescriptor();

    virtual void getSupportedDescriptors() = 0;
    // TODO [DS]: Should be moved into Node derivative class
    virtual void createDescriptor(const std::vector<MKLDNNMemoryDesc>& inputDesc,
                                  const std::vector<MKLDNNMemoryDesc>& outputDesc) {}
    virtual void initDescriptor(const NodeConfig& config);
    virtual bool created() const = 0;
    virtual bool created(const MKLDNNExtensionManager::Ptr& extMgr) {
        return created();
    }

    /**
     * @brief Performs Node initialization based on graph context.
     * This is an auxiliary method that allows to use information not available in Node constructor (e.g. connection information with other nodes)
     */
    virtual void init() {}

    template <class PD, class D, typename FPD = bool>
    PD createPrimitiveDescriptor(const mkldnn::primitive_attr &attr = mkldnn::primitive_attr()) {
        auto descsEqual = [](const std::vector<InferenceEngine::TensorDesc>& srcDescs,
                               const std::vector<InferenceEngine::DataConfig>& selectedDescs) {
            if (srcDescs.empty() && selectedDescs.empty())
                return true;
            if (srcDescs.empty() || selectedDescs.empty())
                return false;
            for (size_t i = 0; i < srcDescs.size() && i < selectedDescs.size(); i++) {
                if (!(srcDescs[i].getBlockingDesc() == selectedDescs[i].desc.getBlockingDesc() &&
                      srcDescs[i].getPrecision() == selectedDescs[i].desc.getPrecision() &&
                      srcDescs[i].getDims() == selectedDescs[i].desc.getDims()) &&
                      srcDescs[i].getLayout() != InferenceEngine::Layout::ANY)
                    return false;
            }
            return true;
        };

        const NodeDesc *selected_pd = getSelectedPrimitiveDescriptor();
        if (selected_pd == nullptr)
            IE_THROW() << "Preferable primitive descriptor is not set for node " << getName() << ".";

        for (const auto& desc : descs) {
            auto itpd = desc.createPrimitiveDescriptorIterator(engine, attr);

            while (static_cast<bool>(itpd))  {
                std::vector<InferenceEngine::TensorDesc> srcDescs;
                for (size_t i = 0; i < descInputNumbers(desc); i++)
                    srcDescs.push_back(*getSrcMemDesc(itpd, i));

                std::vector<InferenceEngine::TensorDesc> dstDescs;
                for (size_t i = 0; i < descOutputNumbers(desc); i++)
                    dstDescs.push_back(*getDstMemDesc(itpd, i));

                impl_desc_type impl_type = parse_impl_name(itpd.impl_info_str());

                if (impl_type == selected_pd->getImplementationType() &&
                    descsEqual(srcDescs, selected_pd->getConfig().inConfs) &&
                    descsEqual(dstDescs, selected_pd->getConfig().outConfs)) {
                    prepareMemory(selected_pd, itpd);
                    PD prim_desc = createPd<PD, D, FPD>(desc);
                    return {itpd.get()};
                }
                if (!itpd.next_impl())
                    break;
            }
        }

        IE_THROW() << "Primitive descriptor was not found for node " << getName() << ".";
    }

    int getExecIndex() const {
        return execIndex;
    }

    std::string getTypeStr() const {
        return typeStr;
    }

    void setTypeStr(const std::string &typeStr) {
        this->typeStr = typeStr;
    }

    virtual size_t descInputNumbers(MKLDNNDescriptor desc) {
        return desc.inputNumbers();
    }

    virtual size_t descOutputNumbers(MKLDNNDescriptor desc) {
        return desc.outputNumbers();
    }

    const PerfCounters & perfCounters() const {
        return profiling;
    }

    PerfCounters & perfCounters() {
        return profiling;
    }

    /**
     * @brief Returns runtime node precision based on input/output data types or data type used for computations
     * @return Runtime node precision
     */
    virtual InferenceEngine::Precision getRuntimePrecision() const;

    const std::vector<InferenceEngine::Precision>& getOriginalInputPrecisions() const {
        return originalInputPrecisions;
    }
    const std::vector<InferenceEngine::Precision>& getOriginalOutputPrecisions() const {
        return originalOutputPrecisions;
    }

    InferenceEngine::Precision getOriginalInputPrecisionAtPort(size_t port) const {
        if (originalInputPrecisions.size() <= port) {
            IE_THROW() << "Incorrect input port number for node " << getName();
        }
        return originalInputPrecisions[port];
    }
    InferenceEngine::Precision getOriginalOutputPrecisionAtPort(size_t port) const {
        if (originalOutputPrecisions.size() <= port) {
            IE_THROW() << "Incorrect output port number for node " << getName();
        }
        return originalOutputPrecisions[port];
    }

    void setOriginalInputPrecisionAtPort(size_t port, InferenceEngine::Precision precision) {
        if (originalInputPrecisions.size() <= port) {
            IE_THROW() << "Incorrect input port number for node " << getName();
        }
        originalInputPrecisions[port] = precision;
    }

    void setOriginalOutputPrecisionAtPort(size_t port, InferenceEngine::Precision precision) {
        if (originalOutputPrecisions.size() <= port) {
            IE_THROW() << "Incorrect output port number for node " << getName();
        }
        originalOutputPrecisions[port] = precision;
    }

    void addOriginalInputPrecision(InferenceEngine::Precision precision) {
        originalInputPrecisions.push_back(precision);
    }

    void addOriginalOutputPrecision(InferenceEngine::Precision precision) {
        originalOutputPrecisions.push_back(precision);
    }

    size_t getOriginalInputsNumber() const {
        return originalInputPrecisions.size();
    }

    size_t getOriginalOutputsNumber() const {
        return originalOutputPrecisions.size();
    }

    Algorithm getAlgorithm() const {
        return algorithm;
    }

    void setAlgorithm(Algorithm alg) {
        algorithm = alg;
    }

    virtual bool canFuse(const MKLDNNNodePtr& node) const {
        return false;
    }

    void setQuantizedGraphFlag(bool flag) {
        isInQuantizedGraph = flag;
    }

    bool canBePerformedAsScaleShift(const MKLDNNNode *parentNode = nullptr) const;

protected:
    bool canFuseSimpleOperation(const MKLDNNNodePtr& node) const;
    // TODO [mandrono]: place outside of the node API
    void fillScalesAndShifts(const MKLDNNNode *parentNode, std::vector<float> &scales, std::vector<float> &shifts, const int align = -1);

    void setType(Type type) {
        this->type = type;
    }

    virtual int getMaxBatch();


    virtual std::unique_ptr<MemoryDesc> getDefinedInputDesc(const NodeConfig &config, size_t idx) const;
    virtual std::unique_ptr<MemoryDesc> getDefinedOutputDesc(const NodeConfig &config, size_t idx) const;
    virtual std::unique_ptr<MKLDNNMemoryDesc> getSrcMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx);
    virtual std::unique_ptr<MKLDNNMemoryDesc> getDstMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx);

    /**
     * @brief Appends new item into ops list with the information on how the node should be executed as post operation.
     * Seed node should call this routine and pass its post operations list as parameter.
     * @param ops List of fused post operations
     */
    virtual void appendPostOps(mkldnn::post_ops& ops);
    virtual std::shared_ptr<mkldnn::primitive_attr> initPrimitiveAttr() const { return nullptr; }

    typedef std::function<MKLDNNMemoryDesc (mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx)>
            GetPrimitiveMemoryFormatFunc;
    std::vector<GetPrimitiveMemoryFormatFunc> internalBlobDesc;

    std::vector<Shape> inputShapes;
    std::vector<Shape> outputShapes;

    std::vector <MKLDNNNodePtr> fusedWith;
    std::vector <MKLDNNNodePtr> mergedWith;
    std::vector <impl_desc_type> implPriorities;
    std::vector <mkldnn::memory::format_tag> inputMemoryFormatsFilter;
    std::vector <mkldnn::memory::format_tag> outputMemoryFormatsFilter;

    std::string originalLayers;  // contains names of the original layers separated by comma

    MKLDNNNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &w_cache);
    MKLDNNNode(const std::string& type, const std::string& name, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &w_cache);

    int selectedPrimitiveDescriptorIndex = -1;
    bool permanent = false;
    bool temporary = false;
    int dynBatchLim = 0;
    enum class ConstantType {
        Unknown,
        Const,
        NoConst
    };
    ConstantType constant = ConstantType::Unknown;
    std::vector<InferenceEngine::Blob::Ptr> internalBlobs;
    std::vector<MKLDNNMemoryPtr> internalBlobMemory;
    std::vector<NodeDesc> supportedPrimitiveDescriptors;
    std::unordered_map<int, mkldnn::memory> primArgs;
    MKLDNNPrimitive prim;
    std::vector<MKLDNNDescriptor> descs;

    InferenceEngine::Blob::Ptr ext_scales;
    MKLDNNWeightsSharing::Ptr weightCache;

    Algorithm algorithm = Algorithm::Undefined;

    bool isInQuantizedGraph = false;

    friend class MKLDNNEdge;
    friend class MKLDNNGraph;
    friend class MKLDNNGraphOptimizer;
    friend class NodeDumper;

    void selectPreferPrimitiveDescriptor(const std::vector<impl_desc_type>& priority, bool ignoreConstInputs);
    bool isConfigDefined(const NodeConfig &config) const;
    virtual bool canBeInPlace() const;

    virtual const std::vector<impl_desc_type>& getPrimitivesPriority();

    virtual std::vector<mkldnn::memory::format_tag> getAvailableFormatsForDims(const MKLDNNDims& dims) const;
    int batchToProcess();

    InferenceEngine::Layout getWeightsLayoutByDims(InferenceEngine::SizeVector dims, bool isGrouped);

    /**
     * @brief Auxiliary function to get node input precisions
     * @return Vector of precisions based on information from node input edges. Return empty vector in case edges are not initialized yet.
     */
    virtual std::vector<InferenceEngine::Precision> getInputPrecisions() const;

    /**
     * @brief Auxiliary function to get node output precisions
     * @return Vector of precisions based on information from node output edges. Return empty vector in case edges are not initialized yet.
     */
    virtual std::vector<InferenceEngine::Precision> getOutputPrecisions() const;

    void addSupportedPrimDesc(const std::vector<PortConfigurator>& inPortConfigs,
                              const std::vector<PortConfigurator>& outPortConfigs,
                              impl_desc_type implType,
                              bool dynBatchSupport = false) {
        auto fill_port = [] (const PortConfigurator& portConfigurator, const Shape& shape,
                             InferenceEngine::Precision prc, std::vector<PortConfig>& port) -> bool {
            // In order to simplify particular node initialization logic we just don't add config in case target shape is not supported by tensorDescCreator.
            // This should be suitable for major of scenarios since almost all nodes add `ncsp` tensorDescCreator which supports any shape rank.
            if (shape.getRank() < portConfigurator.tensorDescCreator->getMinimalRank())
                return false;

            PortConfig portConfig;
            portConfig.inPlace = portConfigurator.inPlace;
            portConfig.constant = portConfigurator.constant;
            portConfig.desc = std::unique_ptr<MemoryDesc>(new BlockedMemoryDesc(portConfigurator.tensorDescCreator->createDesc(prc, shape.getStaticDims())));

            port.push_back(std::move(portConfig));

            return true;
        };

        NodeConfig config;
        for (size_t i = 0; i < inPortConfigs.size(); i++) {
            auto shape = inPortConfigs[i].shape.getRank() == 0 ? getParentEdgesAtPort(i)[0]->getShape() : inPortConfigs[i].shape;
            auto prc = inPortConfigs[i].prc == InferenceEngine::Precision::UNSPECIFIED ? getOriginalInputPrecisionAtPort(i) : inPortConfigs[i].prc;
            if (!fill_port(inPortConfigs[i], shape, prc, config.inConfs))
                return;
        }

        for (size_t i = 0; i < outPortConfigs.size(); i++) {
            auto dims = outPortConfigs[i].shape.getRank() == 0 ? getChildEdgesAtPort(i)[0]->getShape() : outPortConfigs[i].shape;
            auto prc = outPortConfigs[i].prc == InferenceEngine::Precision::UNSPECIFIED ? getOriginalOutputPrecisionAtPort(i) : outPortConfigs[i].prc;
            if (!fill_port(outPortConfigs[i], dims, prc, config.outConfs))
                return;
        }

        config.dynBatchSupport = dynBatchSupport;
        supportedPrimitiveDescriptors.push_back({config, implType});
    }

private:
    std::vector<MKLDNNEdgeWeakPtr> parentEdges;
    std::vector<MKLDNNEdgeWeakPtr> childEdges;

    std::vector<InferenceEngine::Precision> originalInputPrecisions;
    std::vector<InferenceEngine::Precision> originalOutputPrecisions;

    int fusingPort;

    mkldnn::engine engine;

    std::string name;
    std::string typeStr;
    Type type;
    int execIndex = -1;

    std::string typeToStr(Type type);

    PerfCount perfCounter;
    PerfCounters profiling;

    bool isEdgesEmpty(const std::vector<MKLDNNEdgeWeakPtr>& edges) const;

    template <class PD, class D, typename FPD>
    typename std::enable_if<!std::is_same<FPD, bool>::value, PD>::type
    createPd(MKLDNNDescriptor desc) {
        std::shared_ptr<D> selected_desc_ptr = desc;
        std::shared_ptr<FPD> backward_prim_desc_ptr = desc;
        return PD(*selected_desc_ptr, engine, *backward_prim_desc_ptr);
    }

    template <class PD, class D, typename FPD>
    typename std::enable_if<std::is_same<FPD, bool>::value, PD>::type
    createPd(MKLDNNDescriptor desc) {
        std::shared_ptr<D> selected_desc_ptr = desc;
        return PD(*selected_desc_ptr, engine);
    }

    void prepareMemory(const NodeDesc *selected_pd, mkldnn::primitive_desc_iterator& itpd);
    enum LOOK { LOOK_UP = 1, LOOK_DOWN = 2 };
    ConstantType checkConstant(LOOK look, std::vector<MKLDNNNodePtr>& checkNodes);
};

class MKLDNNNode::NodesFactory : public openvino::cc::Factory<Type,
                                            MKLDNNNode*(const std::shared_ptr<ngraph::Node>& op,
                                                        const mkldnn::engine &,
                                                        MKLDNNWeightsSharing::Ptr &)> {
public:
    NodesFactory()
        : Factory("NodesFactory") {}

    MKLDNNNode* create(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng,
                       const MKLDNNExtensionManager::Ptr& extMgr, MKLDNNWeightsSharing::Ptr &w_cache);
};

template<typename MKLDNNNodeType>
struct MKLDNNNodeImpl : public MKLDNNNodeType {
    MKLDNNNodeImpl(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache)
        : MKLDNNNodeType(op, eng, cache) {
        MKLDNNNodeType::perfCounters().template buildClassCounters<MKLDNNNodeType>(NameFromType(MKLDNNNodeType::getType()));
    }
};

#define REG_MKLDNN_CONCAT3_(X, Y, Z) X ## Y ## Z
#define REG_MKLDNN_CONCAT3(X, Y, Z) REG_MKLDNN_CONCAT3_(X, Y, Z)

#define REG_MKLDNN_PRIM_FOR(__prim, __type)                                                 \
static struct REG_MKLDNN_CONCAT3(Registrar4, __prim, __LINE__) {                            \
    REG_MKLDNN_CONCAT3(Registrar4, __prim, __LINE__)() {                                    \
        MKLDNNNode::factory()                                                               \
            .registerNodeIfRequired(MKLDNNPlugin, __prim, __type, MKLDNNNodeImpl<__prim>);  \
    }                                                                                       \
} REG_MKLDNN_CONCAT3(_reg_, __prim, __LINE__);

}  // namespace MKLDNNPlugin
