// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <common/utils.hpp>
#include <oneapi/dnnl/dnnl.hpp>
#include "cpu_memory.h"
#include "cpu_shape.h"
#include "cpu_types.h"
#include "edge.h"
#include "memory_desc/cpu_memory_desc.h"
#include "selective_build.h"
#include "memory_desc/dnnl_memory_desc.h"
#include "onednn/dnnl.h"
#include "onednn/iml_type_mapper.h"
#include <openvino/itt.hpp>
#include "openvino/cc/factory.h"
#include "openvino/core/node.hpp"
#include <nodes/common/blocked_desc_creator.h>
#include "nodes/node_config.h"
#include <shape_inference/shape_inference_cpu.hpp>
#include "perf_count.h"
#include "utils/debug_capabilities.h"
#include "utils/bit_util.hpp"
#include "utils/debug_capabilities.h"

#include "graph_context.h"
#include "nodes/executors/executor.hpp"

#include <memory>
#include <vector>
#include <string>

#define THROW_CPU_NODE_ERR(...) OPENVINO_THROW("[CPU] ", getTypeStr(), " node with name '", getName(), "' ", __VA_ARGS__)
#define CPU_NODE_ASSERT(condition, ...) OPENVINO_ASSERT(condition, getTypeStr(), " node with name '", getName(), "' ", __VA_ARGS__)

namespace ov {
namespace intel_cpu {

using NodePtr = std::shared_ptr<Node>;
using NodeConstPtr = std::shared_ptr<const Node>;
using NodeWeakPtr = std::weak_ptr<Node>;

class PortConfigurator {
public:
    PortConfigurator(ov::intel_cpu::LayoutType blockedDescType, ov::element::Type prc, const Shape& shape,
                     bool constant = false, int inPlace = -1) :
            blockedDescCreator(getBlockedDescCreator(blockedDescType)), prc(prc), shape(shape), constant(constant), inPlace(inPlace) {}

    PortConfigurator(ov::intel_cpu::LayoutType blockedDescType, ov::element::Type prc = ov::element::undefined,
                     bool constant = false, int inPlace = -1) :
            blockedDescCreator(getBlockedDescCreator(blockedDescType)), prc(prc), constant(constant), inPlace(inPlace) {}

    ov::intel_cpu::BlockedDescCreator::CreatorConstPtr blockedDescCreator;
    const ov::element::Type prc;
    const Shape shape;
    bool constant = false;
    int inPlace = -1;

private:
    static ov::intel_cpu::BlockedDescCreator::CreatorConstPtr getBlockedDescCreator(ov::intel_cpu::LayoutType blockedDescType) {
        auto& creators = ov::intel_cpu::BlockedDescCreator::getCommonCreators();
        if (creators.find(blockedDescType) == creators.end()) {
            OPENVINO_THROW("Cannot find tensor descriptor creator");
        }
        return creators.at(blockedDescType);
    }
};

class NodeDesc {
public:
    NodeDesc(NodeConfig conf, impl_desc_type type):
        config(std::move(conf)), implementationType(type), executorFactory(nullptr) {}

    NodeDesc(NodeConfig conf, impl_desc_type type, ExecutorFactoryLegacyPtr factory):
        config(std::move(conf)), implementationType(type), executorFactory(factory) {}

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

    ExecutorFactoryLegacyPtr getExecutorFactory() const {
        return executorFactory;
    }

    template <typename T,
            typename std::enable_if<!std::is_pointer<T>::value && !std::is_reference<T>::value, int>::type = 0,
            typename std::enable_if<std::is_base_of<ExecutorFactoryLegacy, T>::value, int>::type = 0>
    std::shared_ptr<T> getExecutorFactoryAs() {
        auto casted = std::dynamic_pointer_cast<T>(executorFactory);
        if (!casted)
            OPENVINO_THROW("Cannot dynamically cast ExecutorFactory");
        return casted;
    }

    void setExecutorFactory(ExecutorFactoryLegacyPtr factory) {
        executorFactory = factory;
    }

private:
    NodeConfig config;
    impl_desc_type implementationType;
    ExecutorFactoryLegacyPtr executorFactory;
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

    // @todo the method is used when graph is "preconstructed" before creation of the actual graph object
    // remove, as soon edges are added via Graph interface exclusively
    static void addEdge(const EdgePtr& edge);

    virtual void cleanup();
    void remove();

    void addParentEdge(const EdgePtr& edge) {
        assert(std::none_of(parentEdges.begin(), parentEdges.end(),
                            [&edge](const EdgeWeakPtr& _edge){
                                return _edge.lock()->getOutputNum() == edge->getOutputNum();
                            }));
        parentEdges.insert(std::upper_bound(parentEdges.begin(), parentEdges.end(), edge,
                                            [](const EdgeWeakPtr& lhs, const EdgeWeakPtr& rhs) {
                                                return lhs.lock()->getOutputNum() < rhs.lock()->getOutputNum();
                                            }),
                           edge);
        updateConstantType();
    }

    void addChildEdge(const EdgePtr& edge) {
        childEdges.push_back(edge);
    }

    void removeParentEdge(const EdgePtr edge) {
        removeEdge(edge, parentEdges);
        updateConstantType();
    }

    void removeChildEdge(const EdgePtr edge) {
        removeEdge(edge, childEdges);
    }

    const std::vector<EdgeWeakPtr> &getParentEdges() const noexcept {
        return parentEdges;
    }

    const std::vector<EdgeWeakPtr> &getChildEdges() const noexcept {
        return childEdges;
    }

    EdgePtr getParentEdgeAt(size_t idx) const;

    /**
     * Returns all the child edges by input port number.
     *
     * Safe way of getting all the child edges at port.
     * Does not require a vector of the child edges to be sorted.
     * Allocates a storage (vector) to collect the child edges.
     */
    std::vector<EdgePtr> getChildEdgesAtPort(int inputNum) const;

    /**
     * Returns a child edge by index.
     *
     * @attention !!! Can only be used after Graph::SortTopologically is performed !!!
     * Optimized way of accessing a child edge at port.
     * If node contains multiple child edges at port, a random one is returned.
     * Has less overhead in comparison with calling getChildEdgesAtPort(idx)[0].
     * The main use case is accessing Memory from edge with less overhead.
     */
    EdgePtr getChildEdgeAt(size_t idx) const;

    MemoryPtr getSrcMemoryAtPort(size_t idx) const {
        return getParentEdgeAt(idx)->getMemoryPtr();
    }

    MemoryPtr getDstMemoryAtPort(size_t idx) const {
        return getChildEdgeAt(idx)->getMemoryPtr();
    }

    void* getSrcDataAtPort(size_t idx) const {
        return getSrcMemoryAtPort(idx)->getData();
    }

    template<typename T>
    T* getSrcDataAtPortAs(size_t idx) const {
        return getSrcMemoryAtPort(idx)->getDataAs<T>();
    }

    void* getDstDataAtPort(size_t idx) const {
        return getDstMemoryAtPort(idx)->getData();
    }

    template<typename T>
    T* getDstDataAtPortAs(size_t idx) const {
        return getDstMemoryAtPort(idx)->getDataAs<T>();
    }

    int inPlaceInputPort(int portIdx) const;
    int inPlaceOutPort(int portIdx) const;

    bool isDropped() {
        return (isEdgesEmpty(childEdges) && isEdgesEmpty(parentEdges));
    }

    const dnnl::engine& getEngine() const {
        return engine;
    }

    bool isInPlace() const;

    // must be called only after Graph::ResolveEdgeConflicts()
    virtual bool isExecutable() const {
        return !hasEmptyInputTensors();
    }

    enum class ConstantType {
        Const,          // Node is placed in a constant subgraph
        NoConst,        // Node is placed in a non-constant subgraph
        StrictNoConst,  // Node produces non-constant subgraph: this type can't be changed and it does not depend on the parent nodes' ConstantType.
    };
    ConstantType getConstantType() const;
    void updateConstantType();
    bool isConstant();

    // return type int supports return -1 in overloading when channel axis doesn't exist
    virtual int getFusingAxis() const {
        return 1;
    }

    static void appendPostOpArgs(const dnnl::primitive_attr& attr,
                                 std::unordered_map<int, dnnl::memory>& primArgs,
                                 const std::unordered_map<int, MemoryPtr>& postOpsArgs);

    bool isFusedWith(Type type) const;

    virtual void addFusedNode(const NodePtr &fusingNode);

    virtual void fuseInto(NodePtr& parentNode) {
        // The graph supports fusing only of consecutive nodes and some graph logic requires to know through which input port a node was fused into parent one.
        for (size_t i = 0; i < getParentEdges().size(); i++) {
            if (getParentEdgeAt(i)->getParent().get() == parentNode.get()) {
                setFusingPort(i);
                break;
            }
        }

        auto parentFusedNodes = parentNode->getFusedWith();
        if (getFusingPort() < 0 && !parentFusedNodes.empty()) {
            for (size_t i = 0; i < getParentEdges().size(); i++) {
                if (getParentEdgeAt(i)->getParent().get() == parentFusedNodes[parentFusedNodes.size() - 1].get()) {
                    setFusingPort(i);
                    break;
                }
            }
        }

        if (getFusingPort() == -1) {
            OPENVINO_THROW("Cannot determine fusing port between nodes: ", parentNode->getName(), " and ", getName());
        }

        parentNode->addFusedNode(getParentEdgeAt(getFusingPort())->getChild());
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

    const std::string &getParallelDomain() const {
        return parallelDomain;
    }

    Type getType() const {
        return type;
    }

    const std::vector<NodeDesc>& getSupportedPrimitiveDescriptors() const {
        return supportedPrimitiveDescriptors;
    }

    inline const NodeDesc* getSelectedPrimitiveDescriptor() const {
        if (selectedPrimitiveDescriptorIndex < 0 ||
            static_cast<size_t>(selectedPrimitiveDescriptorIndex) >= supportedPrimitiveDescriptors.size())
            return nullptr;
        return &supportedPrimitiveDescriptors[selectedPrimitiveDescriptorIndex];
    }

    inline NodeDesc* getSelectedPrimitiveDescriptor() {
        if (selectedPrimitiveDescriptorIndex < 0 ||
            static_cast<size_t>(selectedPrimitiveDescriptorIndex) >= supportedPrimitiveDescriptors.size())
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
     * @brief Returns parent output memory descriptor from given \p edge
     * must be used after selectOptimalPrimitiveDescriptor stage
     * @param edge
     * @return pointer to parent output memory descriptor with type MemoryDesc
     */
    static MemoryDescPtr getParentOutputMemDesc(const EdgePtr& edge);
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
        if (index < 0 || static_cast<size_t>(index) >= supportedPrimitiveDescriptors.size())
            selectedPrimitiveDescriptorIndex = -1;
        else
            selectedPrimitiveDescriptorIndex = index;

        // Each primitive descriptor has its own InPlace status. So after new primitive descriptor selection
        // we should reset InPlace type to definite new status for node using Node::isInPlace()
        inplace = InPlaceType::Unknown;
    }

    virtual std::string getPrimitiveDescriptorType() const;

    PerfCount &PerfCounter() { return perfCounter; }

    virtual void resolveInPlaceEdges(Edge::LOOK look = Edge::LOOK_BOTH);

    // @todo this supposed to be 'execute + executeImpl' instead of 'executeStatic + execute'
    // but this requires changes in all the nodes. Since moving to a numa node right before an execute
    // is a temprorary solution, do it this way for now.
    void executeStatic(const dnnl::stream strm, int numaId = -1);
    void updateShapes();
    void updateDynamicParams();
    void executeDynamic(dnnl::stream strm, int numaId = -1);
    virtual void redefineOutputMemory(const std::vector<VectorDims> &newShapes);
    void redefineOutputMemory(const size_t port, const VectorDims& new_output_shape);
    bool outputShapeDataDependency() const;

    virtual void initSupportedPrimitiveDescriptors();

    /**
     * @brief Filters supportedPrimitiveDescriptors according to the input layouts specified in inputMemoryFormatsFilter
     * and output layouts specified in outputMemoryFormatsFilter
     */
    void filterSupportedPrimitiveDescriptors();

    virtual void createPrimitive();

    virtual void selectOptimalPrimitiveDescriptor();
    virtual void initOptimalPrimitiveDescriptor();
    void resolveInPlaceDirection();

    virtual void getSupportedDescriptors() = 0;
    // TODO [DS]: Should be moved into Node derivative class
    virtual void createDescriptor(const std::vector<MemoryDescPtr>& inputDesc,
                                  const std::vector<MemoryDescPtr>& outputDesc) {}
    virtual void initDescriptor(const NodeConfig& config);
    virtual bool created() const = 0;

    /**
     * @brief Performs Node initialization based on graph context.
     * This is an auxiliary method that allows to use information not available in Node constructor (e.g. connection information with other nodes)
     */
    virtual void init() {}

    int getExecIndex() const {
        return execIndex;
    }

    const std::string & getTypeStr() const {
        return typeStr;
    }

    void setTypeStr(const std::string &typeStr) {
        this->typeStr = typeStr;
    }

    virtual size_t descInputNumbers() {
        return 1;
    }

    virtual size_t descOutputNumbers() {
        return 1;
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
    virtual ov::element::Type getRuntimePrecision() const;

    const std::vector<ov::element::Type>& getOriginalInputPrecisions() const {
        return originalInputPrecisions;
    }
    const std::vector<ov::element::Type>& getOriginalOutputPrecisions() const {
        return originalOutputPrecisions;
    }

    ov::element::Type getOriginalInputPrecisionAtPort(size_t port) const {
        if (originalInputPrecisions.size() <= port) {
            OPENVINO_THROW("Incorrect input port number for node ", getName());
        }
        return originalInputPrecisions[port];
    }
    ov::element::Type getOriginalOutputPrecisionAtPort(size_t port) const {
        if (originalOutputPrecisions.size() <= port) {
            OPENVINO_THROW("Incorrect output port number for node ", getName());
        }
        return originalOutputPrecisions[port];
    }

    void setOriginalInputPrecisionAtPort(size_t port, ov::element::Type precision) {
        if (originalInputPrecisions.size() <= port) {
            OPENVINO_THROW("Incorrect input port number for node ", getName());
        }
        originalInputPrecisions[port] = precision;
    }

    void setOriginalOutputPrecisionAtPort(size_t port, ov::element::Type precision) {
        if (originalOutputPrecisions.size() <= port) {
            OPENVINO_THROW("Incorrect output port number for node ", getName());
        }
        originalOutputPrecisions[port] = precision;
    }

    void addOriginalInputPrecision(ov::element::Type precision) {
        originalInputPrecisions.push_back(precision);
    }

    void addOriginalOutputPrecision(ov::element::Type precision) {
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

    bool canBePerformedAsScaleShift(const Node *parentNode = nullptr) const;

    bool isDynamicNode() const {
        return isDynamic;
    }

    const Shape& getInputShapeAtPort(size_t port) const {
        if (inputShapes.size() <= port) {
            OPENVINO_THROW("Incorrect input port number for node ", getName());
        }
        return inputShapes[port];
    }

    const Shape& getOutputShapeAtPort(size_t port) const {
        if (outputShapes.size() <= port) {
            OPENVINO_THROW("Incorrect output port number for node ", getName());
        }
        return outputShapes[port];
    }

    const std::vector<MemoryPtr>& getInternalBlobs() const {
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

    void fuseDQScales(const float* scaleData, const size_t scaleSize);
    const std::vector<float>& getDQScales() const {
        return DQScales;
    }
    /**
     * @brief Appends new item into ops list with the information on how the node should be executed as post operation.
     * Seed node should call this routine and pass its post operations list as parameter.
     * @param ops List of fused post operations
     */
    virtual void appendPostOps(dnnl::post_ops& ops, const VectorDims& postOpDims, std::unordered_map<int, MemoryPtr>& postOpsMem, const int channelAxis = 1);
    virtual void appendPostOps(dnnl::post_ops& ops, const VectorDims& postOpDims, std::vector<const void*>& postOpsMem, const int channelAxis = 1);
    virtual bool canBeExecutedInInt8() const {
        OPENVINO_THROW_NOT_IMPLEMENTED("canBeExecutedInInt8 not implemented for node with type ",
                                       NameFromType(getType()));
        return false;
    }
    const bool keepOrigPrecision() const {
        return keepOriginalPrecision;
    }

protected:
    bool canFuseSimpleOperation(const NodePtr& node) const;

    void setType(Type type) {
        this->type = type;
    }

    virtual PortDescBasePtr getConsistentInputDesc(const NodeConfig &config, size_t idx) const;
    virtual PortDescBasePtr getConsistentOutputDesc(const NodeConfig &config, size_t idx) const;
    virtual MemoryDescPtr getSrcMemDesc(const dnnl::primitive_desc &prim_desc, size_t idx) const;
    virtual MemoryDescPtr getDstMemDesc(const dnnl::primitive_desc &prim_desc, size_t idx) const;

    virtual AttrPtr initPrimitiveAttr() { return nullptr; }

    typedef std::function<DnnlMemoryDescPtr (dnnl::primitive_desc& primitive_desc_it, size_t idx)>
            GetPrimitiveMemoryFormatFunc;
    std::vector<GetPrimitiveMemoryFormatFunc> internalBlobDesc;

    std::vector<Shape> inputShapes;
    std::vector<Shape> outputShapes;

    std::vector <NodePtr> fusedWith;
    std::vector <NodePtr> mergedWith;

    int curNumaNode = -1;

    void toNumaNode(int numaID);
    virtual void toNumaNodeImpl(int numaID);

    std::string primitivesPriority;
    std::vector <impl_desc_type> customImplPriorities;
    std::vector <dnnl::memory::format_tag> inputMemoryFormatsFilter;
    std::vector <dnnl::memory::format_tag> outputMemoryFormatsFilter;
    bool enforceBF16evenForGraphTail = false;
    bool keepOriginalPrecision  = false;

    std::string originalLayers;  // contains names of the original layers separated by comma
    std::string parallelDomain;

    Node(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr ctx, const ShapeInferFactory& shapeInferFactory);
    Node(const std::string& type,
         std::vector<Shape> inputShapes,
         std::vector<Shape> outputShapes,
         std::vector<ov::element::Type> originalInputPrecisions,
         std::vector<ov::element::Type> originalOutputPrecisions,
         const std::string& name,
         const GraphContext::CPtr ctx);

    int selectedPrimitiveDescriptorIndex = -1;

    enum class InPlaceType {
        Unknown,
        InPlace,
        NoInPlace
    };
    mutable InPlaceType inplace = InPlaceType::Unknown;
    ConstantType constant = ConstantType::NoConst;
    std::vector<MemoryPtr> internalBlobs;
    std::vector<MemoryPtr> internalBlobMemory;
    std::vector<NodeDesc> supportedPrimitiveDescriptors;
    std::unordered_map<int, dnnl::memory> primArgs;
    std::unordered_map<int, MemoryPtr> postOpsArgs;
    std::vector<dnnl::primitive_desc> descs;

    const GraphContext::CPtr context;

    Algorithm algorithm = Algorithm::Default;

    friend class Edge;
    friend class Graph;
    friend class GraphOptimizer;

    void selectPreferPrimitiveDescriptor(const std::vector<impl_desc_type>& priority, bool ignoreConstInputs);
    void selectPreferPrimitiveDescriptorWithShape(const std::vector<impl_desc_type>& priority, bool ignoreConstInputs);
    bool isOneDimShape(const ov::PartialShape& pshape);
    bool isReorderRequired(ov::intel_cpu::MemoryDescPtr desc1, ov::intel_cpu::MemoryDescPtr desc2);
    bool isConfigDefined(const NodeConfig &config) const;
    virtual bool canBeInPlace() const;

    /* returns default implementaion prioirity */
    virtual const std::vector<impl_desc_type>& getDefaultImplPriority();
    /* returns custom implementation priority + default implementation priority appended as a fallback
     * if custom implementaiton priority is not specified, returns default implementation priority */
    const std::vector<impl_desc_type>& getImplPriority();

    virtual std::vector<dnnl::memory::format_tag> getAvailableFormatsForDims(const Shape& dims) const;

    dnnl::memory::format_tag getWeightsFormatTagByDims(const VectorDims& dims) const;

    /**
     * @brief Auxiliary function to get node input precisions
     * @return Vector of precisions based on information from node input edges. Return empty vector in case edges are not initialized yet.
     */
    virtual std::vector<ov::element::Type> getInputPrecisions() const;

    /**
     * @brief Auxiliary function to get node output precisions
     * @return Vector of precisions based on information from node output edges. Return empty vector in case edges are not initialized yet.
     */
    virtual std::vector<ov::element::Type> getOutputPrecisions() const;

    void addSupportedPrimDesc(const std::vector<PortConfigurator>& inPortConfigs,
                              const std::vector<PortConfigurator>& outPortConfigs,
                              impl_desc_type implType);

    void prepareMemory(const std::vector<DnnlMemoryDescPtr>& intDescs);
    virtual void prepareMemory(const DnnlMemoryDescPtr& intDesc, size_t indx);
    void prepareMemory(dnnl::primitive_desc_iterator& itpd);

    MemoryPtr prepareWeightMemory(DnnlMemoryDescPtr dstWeightDesc, DnnlMemoryDescPtr srcWeightDesc = nullptr);

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
    std::vector<VectorDims> shapeInferGeneric(const std::vector<Shape>& inputDims) const;
    virtual IShapeInfer::Result shapeInfer() const;

    void execute(dnnl::stream stream, int numaId);
    virtual void execute(dnnl::stream strm) = 0;
    // TODO [DS] : make pure after all nodes support dynamic shapes
    virtual void executeDynamicImpl(dnnl::stream strm) {
        OPENVINO_THROW_NOT_IMPLEMENTED("[DS] executeDynamicImpl not implemented for node with type: ", getTypeStr());
    }

    virtual bool needPrepareParams() const;
    // TODO [mandrono]: add description
    // called after memory allocation/reallocation
    virtual void prepareParams() {
        OPENVINO_THROW_NOT_IMPLEMENTED("[DS] prapareParams not implemented for node with type ",
                                       NameFromType(getType()));
    }

    MemoryPtr getScratchPadMem(const MemoryDescPtr& desc) {
        if (!scratchpadMem || !scratchpadMem->getDesc().isCompatible(*desc)) {
            scratchpadMem = context->getScratchPad(curNumaNode)->createScratchPadMem(desc);
        }
        return scratchpadMem;
    }

    std::vector<VectorDims> lastInputDims = {};

    std::shared_ptr<IShapeInfer> shapeInference;

    // we cannot rely on per-NUMA weightCache for caching weights because:
    //   1.it may not exist(in single stream configuration)
    //   2.it only holds weak references, the life-cycle of cached item
    //     is still under control of strong references outside of cache.
    // privateWeightCache is for holding strong references to constant weight
    // copies of same content with different layouts.
    std::shared_ptr<std::unordered_map<std::string, MemoryPtr>> privateWeightCache
    = std::make_shared<std::unordered_map<std::string, MemoryPtr>>();

private:
    static void removeEdge(const EdgePtr edge, std::vector<EdgeWeakPtr> &edges) {
        edges.erase(std::remove_if(edges.begin(), edges.end(),
                                   [&edge] (EdgeWeakPtr _edge) {
                                       return _edge.lock() == edge;
                                   }),
                    edges.end());
    }

    bool isEdgesEmpty(const std::vector<EdgeWeakPtr>& edges) const;

    std::vector<EdgeWeakPtr> parentEdges;
    std::vector<EdgeWeakPtr> childEdges;

    std::vector<ov::element::Type> originalInputPrecisions;
    std::vector<ov::element::Type> originalOutputPrecisions;

    int fusingPort;

    const dnnl::engine engine;

    std::string name;
    std::string typeStr;
    Type type;
    int execIndex = -1;

    std::string typeToStr(Type type);

    PerfCount perfCounter;
    PerfCounters profiling;

    MemoryPtr scratchpadMem;

    // Hold output scales
    std::vector<float> DQScales;

    CPU_DEBUG_CAP_ENABLE(friend class Verbose);
};

#ifndef CPU_DEBUG_CAPS
std::ostream& operator<<(std::ostream&, const Node&);

std::ostream& operator<<(std::ostream&, const Node*);
#endif

template <class... T>
constexpr uint64_t PortMask(T... rest) {
    return util::bit::mask(rest...);
}

class Node::NodesFactory : public openvino::cc::Factory<Type,
                                            Node*(const std::shared_ptr<ov::Node>& op,
                                                  const GraphContext::CPtr)> {
public:
    NodesFactory();

    Node* create(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context);
};

template<typename NodeType>
struct NodeImpl : public NodeType {
    NodeImpl(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context)
        : NodeType(op, context) {
        NodeType::perfCounters().template buildClassCounters<NodeType>(NameFromType(NodeType::getType()));
    }
};

}   // namespace intel_cpu
}   // namespace ov
