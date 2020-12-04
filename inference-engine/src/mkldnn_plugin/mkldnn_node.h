// Copyright (C) 2018-2020 Intel Corporation
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
#include <ie_common.h>
#include "mkldnn_dims.h"
#include "mkldnn_memory.h"
#include "mkldnn_edge.h"
#include "mkldnn_descriptor.h"
#include "mkldnn/iml_type_mapper.h"
#include "mkldnn_extension_mngr.h"
#include "mkldnn_primitive.h"
#include "mkldnn_weights_cache.hpp"
#include "mkldnn.hpp"
#include <openvino/itt.hpp>

namespace MKLDNNPlugin {

using MKLDNNNodePtr = std::shared_ptr<MKLDNNNode>;
using MKLDNNNodeWeakPtr = std::weak_ptr<MKLDNNNode>;

enum Type {
    Unknown,
    Generic,
    Reorder,
    Input,
    Output,
    Convolution,
    Deconvolution,
    Activation,
    Depthwise,
    Lrn,
    Pooling,
    FullyConnected,
    SoftMax,
    Split,
    Concatenation,
    Eltwise,
    Gemm,
    Crop,
    Reshape,
    Tile,
    SimplerNMS,
    ROIPooling,
    BatchNormalization,
    Flatten,
    Pad,
    Permute,
    Copy,
    MemoryOutput,
    MemoryInput,
    RNNCell,
    RNNSeq,
    Quantize,
    BinaryConvolution,
    DeformableConvolution,
    TensorIterator,
    Convert,
    MVN,
    Normalize,
    ScatterUpdate,
    ScatterElementsUpdate,
    ScatterNDUpdate,
    Interpolate,
    ReduceAnd,
    ReduceL1,
    ReduceL2,
    ReduceLogSum,
    ReduceLogSumExp,
    ReduceMax,
    ReduceMean,
    ReduceMin,
    ReduceOr,
    ReduceProd,
    ReduceSum,
    ReduceSumSquare
};

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
        case Activation:
            return "Activation";
        case Lrn:
            return "Lrn";
        case Pooling:
            return "Pooling";
        case FullyConnected:
            return "FullyConnected";
        case Gemm:
            return "Gemm";
        case SoftMax:
            return "SoftMax";
        case Split:
            return "Split";
        case Concatenation:
            return "Concatenation";
        case Depthwise:
            return "Depthwise";
        case Crop:
            return "Crop";
        case Reshape:
            return "Reshape";
        case Tile:
            return "Tile";
        case SimplerNMS:
            return "SimplerNMS";
        case ROIPooling:
            return "ROIPooling";
        case BatchNormalization:
            return "BatchNormalization";
        case Flatten:
            return "Flatten";
        case Pad:
            return "Pad";
        case Permute:
            return "Permute";
        case Copy:
            return "Copy";
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
        case Quantize:
            return "Quantize";
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
        case Normalize:
            return "Normalize";
        case ScatterUpdate:
            return "ScatterUpdate";
        case ScatterElementsUpdate:
            return "ScatterElementsUpdate";
        case ScatterNDUpdate:
            return "ScatterNDUpdate";
        case Interpolate:
            return "Interpolate";
        case ReduceAnd:
            return "ReduceAnd";
        case ReduceL1:
            return "ReduceL1";
        case ReduceL2:
            return "ReduceL2";
        case ReduceLogSum:
            return "ReduceLogSum";
        case ReduceLogSumExp:
            return "ReduceLogSumExp";
        case ReduceMax:
            return "ReduceMax";
        case ReduceMean:
            return "ReduceMean";
        case ReduceMin:
            return "ReduceMin";
        case ReduceOr:
            return "ReduceOr";
        case ReduceProd:
            return "ReduceProd";
        case ReduceSum:
            return "ReduceSum";
        case ReduceSumSquare:
            return "ReduceSumSquare";
        default:
            return "Unknown";
    }
}

class PrimitiveDescInfo {
public:
    PrimitiveDescInfo(const InferenceEngine::LayerConfig conf, impl_desc_type type): config(conf) {
        implementationType = type;
    }

    PrimitiveDescInfo(const InferenceEngine::LayerConfig conf, impl_desc_type type, std::vector<mkldnn::memory::format> outFmts): config(conf) {
        implementationType = type;
        outputLayouts = outFmts;
    }

    PrimitiveDescInfo(const InferenceEngine::LayerConfig conf, impl_desc_type type, mkldnn::memory::format outFmt): config(conf) {
        implementationType = type;

        setOutputLayouts(outFmt);
    }

    PrimitiveDescInfo(const PrimitiveDescInfo &descInfo) = default;
    PrimitiveDescInfo(PrimitiveDescInfo &&descInfo) = default;

    PrimitiveDescInfo &operator=(const PrimitiveDescInfo &descInfo) = default;

    const InferenceEngine::LayerConfig getConfig() const {
        return config;
    }
    InferenceEngine::LayerConfig& getConfig() {
        return config;
    }

    impl_desc_type getImplementationType() const {
        return implementationType;
    }

    const std::vector<mkldnn::memory::format>& getOutputLayouts() const {
        return outputLayouts;
    }

    void setImplementationType(impl_desc_type type) {
        implementationType = type;
    }

    void setOutputLayouts(mkldnn::memory::format outFmt) {
        outputLayouts.clear();

        for (int i = 0; i < config.outConfs.size(); i++) {
            outputLayouts.push_back(outFmt);
        }
    }

private:
    InferenceEngine::LayerConfig config;
    impl_desc_type implementationType;
    std::vector<mkldnn::memory::format> outputLayouts;
};

class MKLDNNNode : public InferenceEngine::details::no_copy {
public:
    class Factory;
    template<typename To>
    class Registrar;

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

    static Factory & factory();

    ~MKLDNNNode() override = default;

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

    void fuseWith(const MKLDNNNodePtr &fuse) {
        fusedWith.push_back(fuse);
    }

    void clearFusedWith() {
        fusedWith.clear();
    }

    void mergeWith(const MKLDNNNodePtr &merge) {
        mergedWith.push_back(merge);
    }

    void addOriginalLayer(const InferenceEngine::CNNLayerPtr &layer);

    const std::vector <MKLDNNNodePtr> &getMergeWith() {
        return mergedWith;
    }

    const std::vector <MKLDNNNodePtr> &getFusedWith() {
        return fusedWith;
    }

    const std::string getName() const {
        return name;
    }

    const std::string getOriginalLayers() const {
        return originalLayers;
    }

    Type getType() const {
        return type;
    }

    const InferenceEngine::CNNLayerPtr &getCnnLayer() const {
        return cnnLayer;
    }

    const std::vector<PrimitiveDescInfo>& getSupportedPrimitiveDescriptors() const {
        return supportedPrimitiveDescriptors;
    }

    inline const PrimitiveDescInfo* getSelectedPrimitiveDescriptor() const {
        if (selectedPrimitiveDescriptorIndex < 0 ||
            selectedPrimitiveDescriptorIndex >= supportedPrimitiveDescriptors.size())
            return nullptr;
        return &supportedPrimitiveDescriptors[selectedPrimitiveDescriptorIndex];
    }

    inline PrimitiveDescInfo* getSelectedPrimitiveDescriptor() {
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
    virtual void createDescriptor(const std::vector<InferenceEngine::TensorDesc>& inputDesc,
                                  const std::vector<InferenceEngine::TensorDesc>& outputDesc) {}
    virtual void initDescriptor(const InferenceEngine::LayerConfig& config);
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

        const PrimitiveDescInfo *selected_pd = getSelectedPrimitiveDescriptor();
        if (selected_pd == nullptr)
            THROW_IE_EXCEPTION << "Preferable primitive descriptor is not set for node " << getName() << ".";

        for (const auto& desc : descs) {
            auto itpd = desc.createPrimitiveDescriptorIterator(engine, attr);

            while (itpd.is_not_end())  {
                std::vector<InferenceEngine::TensorDesc> srcDescs;
                for (size_t i = 0; i < descInputNumbers(desc); i++)
                    srcDescs.push_back(getSrcMemDesc(itpd, i));

                std::vector<InferenceEngine::TensorDesc> dstDescs;
                for (size_t i = 0; i < descOutputNumbers(desc); i++)
                    dstDescs.push_back(getDstMemDesc(itpd, i));

                impl_desc_type impl_type = parse_impl_name(itpd.get_impl_info_str());

                if (impl_type == selected_pd->getImplementationType() &&
                    descsEqual(srcDescs, selected_pd->getConfig().inConfs) &&
                    descsEqual(dstDescs, selected_pd->getConfig().outConfs)) {
                    prepareMemory(selected_pd, itpd);
                    PD prim_desc = createPd<PD, D, FPD>(desc);
                    itpd.getPrimitiveDescriptor(prim_desc);
                    return prim_desc;
                }
                itpd++;
            }
        }

        THROW_IE_EXCEPTION << "Primitive descriptor was not found for node " << getName() << ".";
    }

    static void invertVectorCopyUtoI(const InferenceEngine::PropertyVector<unsigned int>& src, std::vector<ptrdiff_t>& dst) {
        dst.clear();
        for (int i = 1; i <= src.size(); i++) {
            dst.push_back(static_cast<ptrdiff_t>(src[src.size() - i]));
        }
    }

    std::vector<MKLDNNDims> inDims;

    int getExecIndex() const {
        return execIndex;
    }

    std::string getTypeStr() const {
        return typeStr;
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

protected:
    // TODO: It is necessary only in order to avoid modifications of cnnLayers and original topology
    std::vector<MKLDNNDims> outDims;
    void setType(Type type) {
        this->type = type;
    }

    virtual int getMaxBatch();


    virtual InferenceEngine::TensorDesc getConfiguredInputDesc(const InferenceEngine::LayerConfig& config, size_t idx) const;
    virtual InferenceEngine::TensorDesc getConfiguredOutputDesc(const InferenceEngine::LayerConfig& config, size_t idx) const;
    virtual MKLDNNMemoryDesc getSrcMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx);
    virtual MKLDNNMemoryDesc getDstMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx);

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

    std::vector <MKLDNNNodePtr> fusedWith;
    std::vector <MKLDNNNodePtr> mergedWith;
    std::vector <impl_desc_type> implPriorities;
    std::vector <mkldnn_memory_format_t> inputMemoryFormatsFilter;
    std::vector <mkldnn_memory_format_t> outputMemoryFormatsFilter;

    std::string originalLayers;  // contains names of the original layers separated by comma

    MKLDNNNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &w_cache);

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
    std::vector<PrimitiveDescInfo> supportedPrimitiveDescriptors;
    MKLDNNPrimitive prim;
    std::vector<MKLDNNDescriptor> descs;

    InferenceEngine::Blob::Ptr ext_scales;
    MKLDNNWeightsSharing::Ptr weightCache;

    friend class MKLDNNEdge;
    friend class MKLDNNGraph;
    friend class MKLDNNGraphOptimizer;

    bool isUninitTensorDesc(const InferenceEngine::TensorDesc& desc) const;
    bool isInitConfig(const InferenceEngine::LayerConfig& config) const;
    virtual void selectPreferPrimitiveDescriptor(const std::vector<impl_desc_type>& priority);
    virtual bool canBeInPlace() const;

    virtual const std::vector<impl_desc_type>& getPrimitivesPriority();

    std::vector<mkldnn::memory::format> getAvailableFormatsForDims(const MKLDNNDims& dims) const;
    int batchToProcess();

    InferenceEngine::Blob::Ptr createInternalBlob(InferenceEngine::SizeVector dims, bool weights, bool is_grouped = false);

    InferenceEngine::Layout getWeightsLayoutByDims(InferenceEngine::SizeVector dims, bool isGrouped);

private:
    std::vector<MKLDNNEdgeWeakPtr> parentEdges;
    std::vector<MKLDNNEdgeWeakPtr> childEdges;

    InferenceEngine::CNNLayerPtr cnnLayer;
    mkldnn::engine engine;

    std::string name;
    const std::string typeStr;
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

    void prepareMemory(const PrimitiveDescInfo *selected_pd, mkldnn::primitive_desc_iterator& itpd);
    enum LOOK { LOOK_UP = 1, LOOK_DOWN = 2 };
    ConstantType checkConstant(LOOK look, std::vector<MKLDNNNodePtr>& checkNodes);
};

class MKLDNNNode::Factory : InferenceEngine::details::no_copy {
public:
    using builder_t = std::function<MKLDNNNode *(const InferenceEngine::CNNLayerPtr& layer,
        const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &w_cache)>;

    MKLDNNNode* create(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng,
                       const MKLDNNExtensionManager::Ptr& extMgr, MKLDNNWeightsSharing::Ptr &w_cache);

    void registerNode(Type type, builder_t builder);

private:
    using map_t = std::unordered_map<Type, builder_t,
        std::hash<std::underlying_type<MKLDNNPlugin::Type>::type>>;
    map_t builders;
};

template<typename To>
class MKLDNNNode::Registrar {
public:
    explicit Registrar(Type type) {
        MKLDNNNode::factory().registerNode(type,
                [type](const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng,
                    MKLDNNWeightsSharing::Ptr &w_cache) -> MKLDNNNode* {
                    MKLDNNNode *node = new To(layer, eng, w_cache);
                    node->perfCounters().buildClassCounters<To>(NameFromType(type));
                    return node;
                });
    }
};

#define REG_MKLDNN_CONCAT2(X, Y) X ## Y
#define REG_MKLDNN_CONCAT(X, Y) REG_MKLDNN_CONCAT2(X, Y)
#define REG_MKLDNN_PRIM_FOR(__prim, __type) \
static MKLDNNNode::Registrar<__prim> REG_MKLDNN_CONCAT(_reg_, __LINE__)(__type)

template <typename T, typename U>
inline T div_up(const T a, const U b) {
    assert(b);
    return (a + b - 1) / b;
}

template <typename T, typename U>
inline T rnd_up(const T a, const U b) {
    return div_up(a, b) * b;
}

}  // namespace MKLDNNPlugin
