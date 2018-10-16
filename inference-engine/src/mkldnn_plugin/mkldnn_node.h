// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_api.h>
#include <memory>
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <ie_common.h>
#include <ie_profiling.hpp>
#include <caseless.hpp>
#include "mkldnn_dims.h"
#include "mkldnn_memory.h"
#include "mkldnn_edge.h"
#include "mkldnn_descriptor.h"
#include "mkldnn/iml_type_mapper.h"
#include "mkldnn_extension_mngr.h"
#include "mkldnn_primitive.h"

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
    Convolution_Sum,
    Convolution_Activation,
    Convolution_Sum_Activation,
    Activation,
    Depthwise,
    Lrn,
    Pooling,
    FullyConnected,
    SoftMax,
    Split,
    Concatenation,
    Power,
    Eltwise,
    Crop,
    Reshape,
    Tile,
    SimplerNMS,
    ROIPooling,
    BatchNormalization,
    Flatten,
    Permute,
    Copy,
    MemoryOutput,
    MemoryInput,
};

static Type TypeFromName(const std::string type) {
    static caseless_unordered_map<std::string, Type> type_to_name_tbl = {
            { "Unknown", Unknown },
            { "Input", Input },
            { "Const", Input },
            { "Output", Output },
            { "Reorder", Reorder },
            { "Convolution", Convolution },
            { "ReLU", Activation },
            { "ELU", Activation },
            { "Sigmoid", Activation },
            { "Logistic", Activation },
            { "TanH", Activation },
            { "ReLU6", Activation },
            { "Activation", Activation },
            { "ScaleShift", Depthwise },
            { "PReLU", Depthwise },
            { "Clamp", Activation },
            { "Norm", Lrn },
            { "LRN", Lrn },
            { "Pooling", Pooling },
            { "FullyConnected", FullyConnected },
            { "InnerProduct", FullyConnected },
            { "Softmax", SoftMax },
            { "SoftMax", SoftMax },
            { "Split", Split },
            { "Slice", Split },
            { "Concat", Concatenation },
            { "Power", Power },
            { "Deconvolution", Deconvolution },
            { "Eltwise", Eltwise },
            { "Crop", Crop },
            { "Reshape", Reshape },
            { "Tile", Tile },
            { "SimplerNMS", SimplerNMS },
            { "ROIPooling", ROIPooling },
            { "BatchNormalization", BatchNormalization },
            { "Flatten", Flatten },
            { "Permute", Permute },
            { "Copy", Copy },
            { "MemoryInput", MemoryInput},  // for construction from name ctor, arbitrary name is used
            { "Memory", MemoryOutput },  // for construction from layer ctor
    };

    if (type_to_name_tbl.find(type) != type_to_name_tbl.end()) {
        return type_to_name_tbl[type];
    } else {
        return Unknown;
    }
}

class PrimitiveDescInfo {
public:
    PrimitiveDescInfo(const InferenceEngine::LayerConfig conf, impl_desc_type type): config(conf) {
        implementationType = type;
    }
    PrimitiveDescInfo(const InferenceEngine::LayerConfig conf, const char *desc_native_name): config(conf) {
        implementationType = parse_impl_name(desc_native_name);
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

private:
    InferenceEngine::LayerConfig config;
    impl_desc_type implementationType;
};

class MKLDNNNode : public InferenceEngine::details::no_copy {
public:
    static MKLDNNNode* CreateNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng,
                                  const MKLDNNExtensionManager::Ptr& extMgr);

    ~MKLDNNNode() override = default;

    void addEdge(const MKLDNNEdgeWeakPtr& edge, size_t pIndex, size_t cIndex);
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


    bool isDropped() {
        return (isEdgesEmpty(childEdges) && isEdgesEmpty(parentEdges));
    }

    const mkldnn::engine& getEngine() const {
        return engine;
    }

    bool isConstant();

    bool isInplace() const;

    void fuseWith(const MKLDNNNodePtr &fuse) {
        fusedWith.push_back(fuse);
    }

    void mergeWith(const MKLDNNNodePtr &merge) {
        mergedWith.push_back(merge);
    }

    const std::vector <MKLDNNNodePtr> &getMergeWith() {
        return mergedWith;
    }

    const std::string getName() const {
        return name;
    }

    Type getType() const {
        return type;
    }

    const InferenceEngine::CNNLayerPtr &getCnnLayer() {
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

    InferenceEngine::ProfilingTask &GetProfilingTask() { return profilingTask; }

    virtual void setDynamicBatchLim(int lim);

    void resolveNotAllocatedEdges();
    virtual void execute(mkldnn::stream strm);
    virtual void initSupportedPrimitiveDescriptors();
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

    template <class PD, class D, typename FPD = bool>
    PD createPrimitiveDescriptor(const mkldnn::primitive_attr &attr = mkldnn::primitive_attr()) {
        auto descsEqual = [](const std::vector<InferenceEngine::TensorDesc>& srcDescs,
                               const std::vector<InferenceEngine::DataConfig>& selectedDescs) {
            if (srcDescs.empty() && selectedDescs.empty())
                return true;
            if (srcDescs.empty() || selectedDescs.empty())
                return false;
            for (size_t i = 0; i < srcDescs.size() && i < selectedDescs.size(); i++) {
                if (srcDescs[i] != selectedDescs[i].desc && srcDescs[i].getLayout() != InferenceEngine::Layout::ANY)
                    return false;
            }
            return true;
        };

        const PrimitiveDescInfo *selected_pd = getSelectedPrimitiveDescriptor();
        if (selected_pd == nullptr)
            THROW_IE_EXCEPTION << "Preferable primitive descriptor does not set for node " << getName() << ".";

        for (const auto& desc : descs) {
            try {
                mkldnn::primitive_desc_iterator itpd = desc.createPrimitiveDescriptorIterator(engine, attr);
                do {
                    std::vector<InferenceEngine::TensorDesc> srcDescs;
                    for (size_t i = 0; i < desc.inputNumbers(); i++)
                        srcDescs.push_back(getSrcMemDesc(itpd, i));

                    std::vector<InferenceEngine::TensorDesc> dstDescs;
                    for (size_t i = 0; i < desc.outputNumbers(); i++)
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
                } while (itpd.next());
            } catch (std::exception& e) {
                // it throw exception in case of no implementation found
                continue;
            }
        }

        THROW_IE_EXCEPTION << "Primitive descriptor was not found for node " << getName() << ".";
    }

protected:
    // TODO: It is necessary only in order to avoid modifications of cnnLayers and original topology
    std::vector<MKLDNNDims> outDims;
    std::vector<MKLDNNDims> inDims;
    void setType(Type type) {
        this->type = type;
    }

    int getMaxBatch();

    virtual InferenceEngine::TensorDesc getConfiguredInputDesc(const InferenceEngine::LayerConfig& config, size_t idx) const;
    virtual InferenceEngine::TensorDesc getConfiguredOutputDesc(const InferenceEngine::LayerConfig& config, size_t idx) const;
    virtual MKLDNNMemoryDesc getSrcMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx);
    virtual MKLDNNMemoryDesc getDstMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx);

    typedef std::function<MKLDNNMemoryDesc (mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx)>
            GetPrimitiveMemoryFormatFunc;
    std::vector<GetPrimitiveMemoryFormatFunc> internalBlobDesc;

    std::vector <MKLDNNNodePtr> fusedWith;
    std::vector <MKLDNNNodePtr> mergedWith;
    std::vector <impl_desc_type> implPriorities;

    MKLDNNNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng);

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

    InferenceEngine::Blob::Ptr createInternalBlob(InferenceEngine::SizeVector dims, bool weights);

    template<typename To>
    class Register {
    public:
        Register() {
            Registry::RegisterNode(
                Registry::CreatorByLayerFunction([](const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng) -> MKLDNNNode * {
                    return new To(layer, eng); } ) );
        }
    };

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
    InferenceEngine::ProfilingTask profilingTask;

    bool isEdgesEmpty(const std::vector<MKLDNNEdgeWeakPtr>& edges) const;

    class Registry {
    public:
        typedef std::function<MKLDNNNode *(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng)> CreatorByLayerFunction;

        static MKLDNNNode *CreateNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, const MKLDNNExtensionManager::Ptr& extMgr);

        static void RegisterNode(CreatorByLayerFunction f);
    private:
        static std::vector<CreatorByLayerFunction> _dataByLayer;
    };

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
