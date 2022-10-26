// Copyright (C) 2018-2022 Intel Corporation
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
#include "cpu_memory.h"
#include "edge.h"
#include "dnnl_descriptor.h"
#include "selective_build.h"
#include "onednn/dnnl.h"
#include "onednn/iml_type_mapper.h"
#include "extension_mngr.h"
#include "primitive.h"
#include "weights_cache.hpp"
#include <openvino/itt.hpp>
#include "utils/ngraph_utils.hpp"
#include <ngraph/ops.hpp>
#include <ngraph/node.hpp>
#include <ie_precision.hpp>
#include <nodes/common/blocked_desc_creator.h>
#include "cpu_types.h"
#include "cpu_shape.h"
#include "nodes/node_config.h"
#include "cache/multi_cache.h"

#include <utils/shape_inference/static_shape.hpp>
#include <utils/shape_inference/shape_inference.hpp>
#include "utils/debug_capabilities.h"

namespace ov {
namespace intel_cpu {

using NodePtr = std::shared_ptr<Node>;
using NodeConstPtr = std::shared_ptr<const Node>;
using NodeWeakPtr = std::weak_ptr<Node>;

class PortConfigurator {
public:
    PortConfigurator(ov::intel_cpu::LayoutType blockedDescType, InferenceEngine::Precision prc, const Shape& shape,
                     bool constant = false, int inPlace = -1) :
            blockedDescCreator(getBlockedDescCreator(blockedDescType)), prc(prc), shape(shape), constant(constant), inPlace(inPlace) {}

    PortConfigurator(ov::intel_cpu::LayoutType blockedDescType, InferenceEngine::Precision prc = InferenceEngine::Precision::UNSPECIFIED,
                     bool constant = false, int inPlace = -1) :
            blockedDescCreator(getBlockedDescCreator(blockedDescType)), prc(prc), constant(constant), inPlace(inPlace) {}

    ov::intel_cpu::BlockedDescCreator::CreatorConstPtr blockedDescCreator;
    const InferenceEngine::Precision prc;
    const Shape shape;
    bool constant = false;
    int inPlace = -1;

private:
    static ov::intel_cpu::BlockedDescCreator::CreatorConstPtr getBlockedDescCreator(ov::intel_cpu::LayoutType blockedDescType) {
        auto& creators = ov::intel_cpu::BlockedDescCreator::getCommonCreators();
        if (creators.find(blockedDescType) == creators.end()) {
            IE_THROW() << "Cannot find tensor descriptor creator";
        }
        return creators.at(blockedDescType);
    }
};

class NodeDesc {
public:
    NodeDesc(const NodeConfig& conf, impl_desc_type type): config(conf) {
        implementationType = type;
    }

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

class Node {
public:
    Node(const Node &) = delete;
    Node & operator = (const Node &) = delete;

    using AttrPtr = std::shared_ptr<dnnl::primitive_attr>;

public:
    template<typename T, int N>
    struct Tag {};

    struct PerfCounters {
        PerfCounters(std::string const& name)
            : execute(openvino::itt::handle(name))
            , getSupportedDescriptors(openvino::itt::handle<Tag<Node, 0>>("Node::getSupportedDescriptors"))
            , initSupportedPrimitiveDescriptors(openvino::itt::handle<Tag<Node, 1>>("Node::initSupportedPrimitiveDescriptors"))
            , filterSupportedPrimitiveDescriptors(openvino::itt::handle<Tag<Node, 2>>("Node::filterSupportedPrimitiveDescriptors"))
            , selectOptimalPrimitiveDescriptor(openvino::itt::handle<Tag<Node, 3>>("Node::selectOptimalPrimitiveDescriptor"))
            , createPrimitive(openvino::itt::handle<Tag<Node, 4>>("Node::createPrimitive"))
            , initOptimalPrimitiveDescriptor(openvino::itt::handle<Tag<Node, 5>>("Node::initOptimalPrimitiveDescriptor"))
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

    virtual ~Node() = default;

    void addEdge(const EdgeWeakPtr& edge);
    void removeEdge(const EdgeWeakPtr& edge);

    virtual void cleanup();
    void remove();

    const std::vector<EdgeWeakPtr> &getParentEdges() const noexcept {
        return parentEdges;
    }

    const std::vector<EdgeWeakPtr> &getChildEdges() const noexcept {
        return childEdges;
    }

    const EdgePtr getParentEdgeAt(size_t idx) const;
    virtual const EdgePtr getChildEdgeAt(size_t idx) const;

    const std::vector<EdgePtr> getParentEdgesAtPort(size_t idx) const;
    const std::vector<EdgePtr> getChildEdgesAtPort(size_t idx) const;

    bool isDropped() {
        return (isEdgesEmpty(childEdges) && isEdgesEmpty(parentEdges));
    }

    const dnnl::engine& getEngine() const {
        return engine;
    }

    bool isInPlace();

    // must be called only after Graph::InitEdges()
    virtual bool isExecutable() const {
        return !hasEmptyInputTensors();
    }

    bool isConstant();

    // return type int supports return -1 in overloading when channel axis doesn't exist
    virtual int getFusingAxis() const {
        return 1;
    }

    static void appendPostOpArgs(const dnnl::primitive_attr& attr,
                                 std::unordered_map<int, dnnl::memory>& primArgs,
                                 const std::vector<MemoryPtr>& postOpsArgs);

    bool isFusedWith(Type type) const;

    virtual void addFusedNode(const NodePtr &fusingNode);

    virtual void fuseInto(NodePtr& parentNode) {
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

    void mergeWith(const NodePtr &merge) {
        mergedWith.push_back(merge);
    }

    const std::vector <NodePtr> &getMergeWith() {
        return mergedWith;
    }

    const std::vector <NodePtr> &getFusedWith() {
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

    /**
     * @brief Returns input selected primitive descriptor on the specified port
     * must be used after selectOptimalPrimitiveDescriptor stage
     * @param portNum port number
     * @return pointer to selected primitive descriptor with type MemoryDesc
     */
    MemoryDescPtr getBaseMemDescAtInputPort(size_t portNum) const;

    /**
     * @brief Returns output selected primitive descriptor on the specified port
     * must be used after selectOptimalPrimitiveDescriptor stage
     * @param portNum port number
     * @return pointer to selected primitive descriptor with type MemoryDesc
     */
    MemoryDescPtr getBaseMemDescAtOutputPort(size_t portNum) const;

    /**
     * @brief Returns input selected primitive descriptor on the specified port
     * must be used after selectOptimalPrimitiveDescriptor stage
     * @param portNum port number
     * @return pointer to selected primitive descriptor with type T
     */
    template <typename T,
              typename std::enable_if<!std::is_pointer<T>::value && !std::is_reference<T>::value, int>::type = 0,
              typename std::enable_if<std::is_base_of<MemoryDesc, T>::value, int>::type = 0>
    std::shared_ptr<T> getInputMemDescAtPort(size_t portNum) const;

    /**
     * @brief Returns output selected primitive descriptor on the specified port
     * must be used after selectOptimalPrimitiveDescriptor stage
     * @param portNum port number
     * @return pointer to selected primitive descriptor with type T
     */
    template <typename T,
              typename std::enable_if<!std::is_pointer<T>::value && !std::is_reference<T>::value, int>::type = 0,
              typename std::enable_if<std::is_base_of<MemoryDesc, T>::value, int>::type = 0>
    std::shared_ptr<T> getOutputMemDescAtPort(size_t portNum) const;

    void selectPrimitiveDescriptorByIndex(int index) {
        if (index < 0 || index >= supportedPrimitiveDescriptors.size())
            selectedPrimitiveDescriptorIndex = -1;
        else
            selectedPrimitiveDescriptorIndex = index;

        // Each primitive descriptor has its own InPlace status. So after new primitive descriptor selection
        // we should reset InPlace type to definite new status for node using Node::isInPlace()
        inplace = InPlaceType::Unknown;
    }

    std::string getPrimitiveDescriptorType();

    PerfCount &PerfCounter() { return perfCounter; }

    virtual void setDynamicBatchLim(int lim);

    void resolveInPlaceEdges();

    virtual void execute(dnnl::stream strm);
    void executeDynamic(dnnl::stream strm);
    virtual void redefineOutputMemory(const std::vector<VectorDims> &newShapes);

    virtual void initSupportedPrimitiveDescriptors();

    /**
     * @brief Filters supportedPrimitiveDescriptors according to the input layouts specified in inputMemoryFormatsFilter
     * and output layouts specified in outputMemoryFormatsFilter
     */
    virtual void filterSupportedPrimitiveDescriptors();

    virtual void createPrimitive();

    virtual void selectOptimalPrimitiveDescriptor();
    virtual void initOptimalPrimitiveDescriptor();

    virtual void getSupportedDescriptors() = 0;
    // TODO [DS]: Should be moved into Node derivative class
    virtual void createDescriptor(const std::vector<MemoryDescPtr>& inputDesc,
                                  const std::vector<MemoryDescPtr>& outputDesc) {}
    virtual void initDescriptor(const NodeConfig& config);
    virtual bool created() const = 0;
    virtual bool created(const ExtensionManager::Ptr& extMgr) {
        return created();
    }

    /**
     * @brief Performs Node initialization based on graph context.
     * This is an auxiliary method that allows to use information not available in Node constructor (e.g. connection information with other nodes)
     */
    virtual void init() {}

    template <class PD, class D, typename FPD = bool>
    PD createPrimitiveDescriptor(const dnnl::primitive_attr &attr = dnnl::primitive_attr()) {
        auto descsCompatible = [](const std::vector<MemoryDescPtr>& srcDescs,
                               const std::vector<PortConfig>& selectedDescs) {
            if (srcDescs.empty() && selectedDescs.empty())
                return true;
            if (srcDescs.empty() || selectedDescs.empty())
                return false;
            for (size_t i = 0; i < srcDescs.size() && i < selectedDescs.size(); i++) {
                if (!srcDescs[i]->isCompatible(*selectedDescs[i].getMemDesc()))
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
                std::vector<MemoryDescPtr> srcDescs;
                for (size_t i = 0; i < descInputNumbers(desc); i++)
                    srcDescs.push_back(getSrcMemDesc(itpd, i));

                std::vector<MemoryDescPtr> dstDescs;
                for (size_t i = 0; i < descOutputNumbers(desc); i++)
                    dstDescs.push_back(getDstMemDesc(itpd, i));

                impl_desc_type impl_type = parse_impl_name(itpd.impl_info_str());

                if (impl_type == selected_pd->getImplementationType() &&
                    descsCompatible(srcDescs, selected_pd->getConfig().inConfs) &&
                    descsCompatible(dstDescs, selected_pd->getConfig().outConfs)) {
                    prepareMemory(itpd);
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

    const std::string & getTypeStr() const {
        return typeStr;
    }

    void setTypeStr(const std::string &typeStr) {
        this->typeStr = typeStr;
    }

    virtual size_t descInputNumbers(DnnlDesriptor desc) {
        return desc.inputNumbers();
    }

    virtual size_t descOutputNumbers(DnnlDesriptor desc) {
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

    // TODO: alighn behaviour for original(Input/Output)Precisions and (input/output)Shapes
    /**
     * @brief Returns inputs number which have ngraph nodes.
     * Inputs number compute as size of originalInputPrecisions vector
     * IMPORTANT!!!
     * FuseConvolutionAndBias and FuseMultiplyAndAdd change originalInputPrecisions vector
     * @return original inputs number
     */
    size_t getOriginalInputsNumber() const {
        return originalInputPrecisions.size();
    }

    /**
     * @brief Returns outputs number which have ngraph nodes.
     * Outputs number compute as size of originalOutputPrecisions vector
     * @return original outputs number
     */
    size_t getOriginalOutputsNumber() const {
        return originalOutputPrecisions.size();
    }

    Algorithm getAlgorithm() const {
        return algorithm;
    }

    void setAlgorithm(Algorithm alg) {
        algorithm = alg;
    }

    virtual bool canFuse(const NodePtr& node) const {
        return false;
    }

    void setQuantizedGraphFlag(bool flag) {
        isInQuantizedGraph = flag;
    }

    bool canBePerformedAsScaleShift(const Node *parentNode = nullptr) const;

    bool isDynamicNode() const {
        return isDynamic;
    }

    const Shape& getInputShapeAtPort(size_t port) const {
        if (inputShapes.size() <= port) {
            IE_THROW() << "Incorrect input port number for node " << getName();
        }
        return inputShapes[port];
    }

    const Shape& getOutputShapeAtPort(size_t port) const {
        if (outputShapes.size() <= port) {
            IE_THROW() << "Incorrect output port number for node " << getName();
        }
        return outputShapes[port];
    }

    const std::vector<InferenceEngine::Blob::Ptr>& getInternalBlobs() const {
        return internalBlobs;
    }

    /**
    * @brief Return scales and shift if nodes can be executed as ScaleShift, else raise exception
    * If node has only scale or shift value, fill missing value with default values
    * i.e. EltwiseAdd: fill shifts from constant, fill scales with default values = 1.0f
    * @param parentNode
    * node from which data comes
    * @return pair of scales and shifts
    */
    std::pair<std::vector<float>, std::vector<float>> getScalesAndShifts(const Node *parentNode) const;

    /**
     * @brief Appends new item into ops list with the information on how the node should be executed as post operation.
     * Seed node should call this routine and pass its post operations list as parameter.
     * @param ops List of fused post operations
     */
    virtual void appendPostOps(dnnl::post_ops& ops, const VectorDims& postOpDims, std::vector<MemoryPtr>& postOpsMem, const int channelAxis = 1);
    virtual void appendPostOps(dnnl::post_ops& ops, const VectorDims& postOpDims, std::vector<const void*>& postOpsMem, const int channelAxis = 1);

    virtual void appendBinPostOps(dnnl::post_ops& ops, const VectorDims& postOpDims, std::vector<MemoryPtr>& binaryPostOpsMem);

    void setRuntimeCache(MultiCachePtr cache) {
        rtParamsCache = cache;
    }

    void setSharedMutex(const std::shared_ptr<std::mutex>& mutex) {
        sharedMutex = mutex;
    }

protected:
    bool canFuseSimpleOperation(const NodePtr& node) const;

    void setType(Type type) {
        this->type = type;
    }

    virtual size_t getMaxBatch() const;


    virtual PortDescBasePtr getConsistentInputDesc(const NodeConfig &config, size_t idx) const;
    virtual PortDescBasePtr getConsistentOutputDesc(const NodeConfig &config, size_t idx) const;
    virtual MemoryDescPtr getSrcMemDesc(dnnl::primitive_desc_iterator &primitive_desc_it, size_t idx);
    virtual MemoryDescPtr getDstMemDesc(dnnl::primitive_desc_iterator &primitive_desc_it, size_t idx);

    virtual AttrPtr initPrimitiveAttr() { return nullptr; }

    typedef std::function<DnnlMemoryDescPtr (dnnl::primitive_desc_iterator &primitive_desc_it, size_t idx)>
            GetPrimitiveMemoryFormatFunc;
    std::vector<GetPrimitiveMemoryFormatFunc> internalBlobDesc;

    std::vector<Shape> inputShapes;
    std::vector<Shape> outputShapes;

    std::vector <NodePtr> fusedWith;
    std::vector <NodePtr> mergedWith;
    std::vector <impl_desc_type> implPriorities;
    std::vector <dnnl::memory::format_tag> inputMemoryFormatsFilter;
    std::vector <dnnl::memory::format_tag> outputMemoryFormatsFilter;
    bool enforceBF16evenForGraphTail = false;

    std::string originalLayers;  // contains names of the original layers separated by comma

    Node(const std::shared_ptr<ngraph::Node>& op, const dnnl::engine& eng, WeightsSharing::Ptr &w_cache);
    Node(const std::string& type, const std::string& name, const dnnl::engine& eng, WeightsSharing::Ptr &w_cache);

    int selectedPrimitiveDescriptorIndex = -1;
    bool permanent = false;
    bool temporary = false;
    int dynBatchLim = 0;
    enum class InPlaceType {
        Unknown,
        InPlace,
        NoInPlace
    };
    enum class ConstantType {
        Unknown,
        Const,
        NoConst
    };
    InPlaceType inplace = InPlaceType::Unknown;
    ConstantType constant = ConstantType::Unknown;
    std::vector<InferenceEngine::Blob::Ptr> internalBlobs;
    std::vector<MemoryPtr> internalBlobMemory;
    std::vector<NodeDesc> supportedPrimitiveDescriptors;
    std::unordered_map<int, dnnl::memory> primArgs;
    std::vector<MemoryPtr> postOpsArgs;
    Primitive prim;
    std::vector<DnnlDesriptor> descs;

    WeightsSharing::Ptr weightCache;

    Algorithm algorithm = Algorithm::Default;

    bool isInQuantizedGraph = false;

    friend class Edge;
    friend class Graph;
    friend class GraphOptimizer;

    void selectPreferPrimitiveDescriptor(const std::vector<impl_desc_type>& priority, bool ignoreConstInputs);
    bool isConfigDefined(const NodeConfig &config) const;
    virtual bool canBeInPlace() const;

    virtual const std::vector<impl_desc_type>& getPrimitivesPriority();

    virtual std::vector<dnnl::memory::format_tag> getAvailableFormatsForDims(const Shape& dims) const;
    int batchToProcess() const;

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
            // In order to simplify particular node initialization logic we just don't add config in case target shape is not supported by blockedDescCreator.
            // This should be suitable for major of scenarios since almost all nodes add `ncsp` blockedDescCreator which supports any shape rank.
            if (shape.getRank() < portConfigurator.blockedDescCreator->getMinimalRank())
                return false;

            PortConfig portConfig;
            portConfig.inPlace(portConfigurator.inPlace);
            portConfig.constant(portConfigurator.constant);
            portConfig.setMemDesc(portConfigurator.blockedDescCreator->createSharedDesc(prc, shape));

            port.push_back(std::move(portConfig));

            return true;
        };

        NodeConfig config;
        for (size_t i = 0; i < inPortConfigs.size(); i++) {
            auto shape = inPortConfigs[i].shape.getRank() == 0 ? getInputShapeAtPort(i) : inPortConfigs[i].shape;
            auto prc = inPortConfigs[i].prc == InferenceEngine::Precision::UNSPECIFIED ? getOriginalInputPrecisionAtPort(i) : inPortConfigs[i].prc;
            if (!fill_port(inPortConfigs[i], shape, prc, config.inConfs))
                return;
        }

        for (size_t i = 0; i < outPortConfigs.size(); i++) {
            auto dims = outPortConfigs[i].shape.getRank() == 0 ? getOutputShapeAtPort(i) : outPortConfigs[i].shape;
            auto prc = outPortConfigs[i].prc == InferenceEngine::Precision::UNSPECIFIED ? getOriginalOutputPrecisionAtPort(i) : outPortConfigs[i].prc;
            if (!fill_port(outPortConfigs[i], dims, prc, config.outConfs))
                return;
        }

        config.dynBatchSupport = dynBatchSupport;
        supportedPrimitiveDescriptors.push_back({config, implType});
    }

    void prepareMemory(const std::vector<DnnlMemoryDescPtr>& intDescs);
    void prepareMemory(dnnl::primitive_desc_iterator& itpd);

    bool isDynamic = false;

    bool isInputTensorAtPortEmpty(size_t port) const;
    bool isOutputTensorAtPortEmpty(size_t port) const;

    bool hasEmptyInputTensors() const;
    bool hasEmptyOutputTensors() const;

    bool inputShapesDefined() const;
    bool outputShapesDefined() const;
    bool shapesDefined() const;
    void updateLastInputDims();

    bool inputShapesModified() const;
    virtual bool needShapeInfer() const;
    std::vector<VectorDims> shapeInferGeneric(const std::vector<Shape>& inputDims, uint32_t value_port_mask = 0) const;
    std::vector<VectorDims> shapeInferGeneric(uint32_t value_port_mask = 0) const;
    virtual std::vector<VectorDims> shapeInfer() const;
    // TODO [DS] : make pure after all nodes will be support dynamic shapes
    virtual void executeDynamicImpl(dnnl::stream strm) {
        IE_THROW(NotImplemented) << "[DS] executeDynamicImpl not implemented for node with type: " << getTypeStr();
    }

    virtual bool needPrepareParams() const;
    // TODO [mandrono]: add description
    // called after memory allocation/reallocation
    virtual void prepareParams() {
        IE_THROW(NotImplemented) << "[DS] prapareParams not implemented for node with type " << NameFromType(getType());
    }

    MultiCachePtr getRuntimeCache() const {
        return rtParamsCache;
    }

    std::vector<VectorDims> lastInputDims = {};

    std::shared_ptr<IShapeInfer> shapeInference;

    std::shared_ptr<std::mutex> sharedMutex = nullptr;

private:
    std::vector<EdgeWeakPtr> parentEdges;
    std::vector<EdgeWeakPtr> childEdges;

    std::vector<InferenceEngine::Precision> originalInputPrecisions;
    std::vector<InferenceEngine::Precision> originalOutputPrecisions;

    int fusingPort;

    dnnl::engine engine;

    std::string name;
    std::string typeStr;
    Type type;
    int execIndex = -1;

    std::string typeToStr(Type type);

    PerfCount perfCounter;
    PerfCounters profiling;

    MultiCachePtr rtParamsCache;

    bool isEdgesEmpty(const std::vector<EdgeWeakPtr>& edges) const;

    template <class PD, class D, typename FPD>
    typename std::enable_if<!std::is_same<FPD, bool>::value, PD>::type
    createPd(DnnlDesriptor desc) {
        std::shared_ptr<D> selected_desc_ptr = desc;
        std::shared_ptr<FPD> backward_prim_desc_ptr = desc;
        return PD(*selected_desc_ptr, engine, *backward_prim_desc_ptr);
    }

    template <class PD, class D, typename FPD>
    typename std::enable_if<std::is_same<FPD, bool>::value, PD>::type
    createPd(DnnlDesriptor desc) {
        std::shared_ptr<D> selected_desc_ptr = desc;
        return PD(*selected_desc_ptr, engine);
    }

    enum LOOK { LOOK_UP = 1, LOOK_DOWN = 2 };
    ConstantType checkConstant(LOOK look, std::vector<NodePtr>& checkNodes);

    std::vector<VectorDims> shapeInferGeneric(const std::vector<StaticShape>& input_shapes,
                                              uint32_t input_value_port_mask) const;

#ifdef CPU_DEBUG_CAPS
    friend class Verbose;
#endif
};

constexpr uint64_t PortMask(int n) {
    return static_cast<uint64_t>(1) << n;
}

template <class... T>
constexpr uint64_t PortMask(int n, T... rest) {
    return PortMask(rest...) | (1 << n);
}

class Node::NodesFactory : public openvino::cc::Factory<Type,
                                            Node*(const std::shared_ptr<ngraph::Node>& op,
                                                  const dnnl::engine &,
                                                  WeightsSharing::Ptr &)> {
public:
    NodesFactory();

    Node* create(const std::shared_ptr<ngraph::Node>& op, const dnnl::engine& eng,
                 const ExtensionManager::Ptr& extMgr, WeightsSharing::Ptr &w_cache);
};

template<typename NodeType>
struct NodeImpl : public NodeType {
    NodeImpl(const std::shared_ptr<ngraph::Node>& op, const dnnl::engine& eng, WeightsSharing::Ptr &cache)
        : NodeType(op, eng, cache) {
        NodeType::perfCounters().template buildClassCounters<NodeType>(NameFromType(NodeType::getType()));
    }
};

}   // namespace intel_cpu
}   // namespace ov
