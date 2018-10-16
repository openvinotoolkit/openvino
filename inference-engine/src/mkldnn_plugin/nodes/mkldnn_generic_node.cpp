// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <mkldnn_extension_mngr.h>
#include <mkldnn_extension_utils.h>
#include "mkldnn_generic_node.h"
#include <vector>
#include <string>
#include <blob_factory.hpp>

using namespace mkldnn;
using namespace MKLDNNPlugin;

MKLDNNGenericNode::MKLDNNGenericNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng) : MKLDNNNode(layer, eng) {}

void MKLDNNGenericNode::getSupportedDescriptors() {
    if (!genericPrimitive && !extFactory) {
        std::string type = getCnnLayer() ? getCnnLayer()->type : "Generic";
        THROW_IE_EXCEPTION << "Cannot get generic primitive for layer: " << getName() << " with type: " << type;
    }
    if (genericPrimitive && extFactory) {
        extFactory.reset();
    }
}

void MKLDNNGenericNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    InferenceEngine::Precision precision = getCnnLayer()->insData[0].lock()->getPrecision();
    if (precision != InferenceEngine::Precision::FP32)
        precision = InferenceEngine::Precision::FP32;
    auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);
    precision = getCnnLayer()->outData[0]->getPrecision();
    if (precision != InferenceEngine::Precision::FP32)
        precision = InferenceEngine::Precision::FP32;
    auto outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);

    if (genericPrimitive) {
        std::vector<InferenceEngine::MKLDNNPlugin::MKLDNNGenericFormats> formats = genericPrimitive->GetSupportedFormats();
        if (formats.empty())
            THROW_IE_EXCEPTION << "External primitive doesn't have supported formats";
        auto createAllDesc = [](const std::vector<MKLDNNEdgeWeakPtr> & edges,
                const std::vector<InferenceEngine::MKLDNNPlugin::MemoryFormat>& formats) {
            if (formats.size() != 1 || edges.size() < 2)
                return false;
            auto firstDims = edges[0].lock()->getDims();
            for (size_t i = 1; i < edges.size(); i++) {
                if (firstDims != edges[i].lock()->getDims())
                    return true;
            }
            return false;
        };
        for (auto &format : formats) {
            bool isAny = false;
            bool isNotAny = false;
            InferenceEngine::LayerConfig config;
            config.dynBatchSupport = false;
            bool isCompatible = true;
            bool allDescCreate = createAllDesc(getParentEdges(), format.GetInputs());
            for (size_t i = 0; i < getParentEdges().size(); i++) {
                auto input_format = format.GetInputs()[0];
                if (format.GetInputs().size() > i)
                    input_format = format.GetInputs()[i];
                else if (!allDescCreate)
                    break;
                if (!MKLDNNMemory::isConsistant(getParentEdgeAt(i)->getDims(),
                                                MKLDNNExtensionUtils::MemoryFormatToMKLFormat(input_format))) {
                    isCompatible = false;
                    break;
                }
                mkldnn::memory::format mkldnnFormat = MKLDNNExtensionUtils::MemoryFormatToMKLFormat(input_format);
                InferenceEngine::DataConfig dataConf;
                dataConf.inPlace = -1;
                dataConf.constant = false;
                dataConf.desc = MKLDNNMemoryDesc(getParentEdgeAt(i)->getDims(), inputDataType, mkldnnFormat);
                config.inConfs.push_back(dataConf);
                if (dataConf.desc.getLayout() == InferenceEngine::Layout::ANY) {
                    isAny = true;
                } else {
                    isNotAny = true;
                }
            }
            if (isAny && isNotAny) {
                THROW_IE_EXCEPTION << "Layer " << getName() << " has incorrect input formats "
                                   << " (any and not any formats don't supported in the same time).";
            }
            isAny = false;
            isNotAny = false;
            allDescCreate = createAllDesc(getChildEdges(), format.GetOutputs());
            for (size_t i = 0; i < getChildEdges().size(); i++) {
                auto output_format = format.GetOutputs()[0];
                if (format.GetOutputs().size() > i)
                    output_format = format.GetOutputs()[i];
                else if (!allDescCreate)
                    break;
                if (!MKLDNNMemory::isConsistant(getChildEdgeAt(i)->getDims(),
                                                MKLDNNExtensionUtils::MemoryFormatToMKLFormat(
                                                        output_format))) {
                    isCompatible = false;
                    break;
                }
                mkldnn::memory::format mkldnnFormat = MKLDNNExtensionUtils::MemoryFormatToMKLFormat(output_format);
                InferenceEngine::DataConfig dataConf;
                dataConf.inPlace = -1;
                dataConf.constant = false;
                dataConf.desc = MKLDNNMemoryDesc(getChildEdgeAt(i)->getDims(), outputDataType, mkldnnFormat);
                config.outConfs.push_back(dataConf);
                if (dataConf.desc.getLayout() == InferenceEngine::Layout::ANY) {
                    isAny = true;
                } else {
                    isNotAny = true;
                }
            }
            if (isAny && isNotAny) {
                THROW_IE_EXCEPTION << "Layer " << getName() << " has incorrect output formats "
                                   << " (any and not any formats don't supported in the same time).";
            }
            if (isCompatible) {
                supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown);
            }
        }
    } else if (extFactory) {
        InferenceEngine::ResponseDesc resp;
        InferenceEngine::StatusCode rc = extFactory->getImplementations(impls, &resp);
        if (rc != InferenceEngine::OK) {
            THROW_IE_EXCEPTION << resp.msg;
        }
        for (auto &impl : impls) {
            std::vector<InferenceEngine::LayerConfig> configs;
            rc = impl->getSupportedConfigurations(configs, &resp);
            if (rc != InferenceEngine::OK) {
                THROW_IE_EXCEPTION << resp.msg;
            }

            for (auto& config : configs) {
                supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown);
            }
        }
        if (impls.empty()) {
            THROW_IE_EXCEPTION << "Layer " << getName() << " hasn't available configurations!";
        }
    } else {
        THROW_IE_EXCEPTION << "Descriptor for generic primitive doesn't exist";
    }
}

void MKLDNNGenericNode::createPrimitive() {
    if (extFactory) {
        return;
    }
    if (!genericPrimitive)
        THROW_IE_EXCEPTION << "Descriptor for generic primitive doesn't exist";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_IE_EXCEPTION << "Preferable primitive descriptor does not set.";
}

void MKLDNNGenericNode::execute(mkldnn::stream strm) {
    if (genericPrimitive) {
        for (size_t i = 0; i < getParentEdges().size(); i++) {
            auto& mklMemory = getParentEdgeAt(i)->getMemory();
            inputs.push_back(MKLDNNExtensionUtils::MKLMemoryToGenericMemory(mklMemory));
        }

        for (size_t i = 0; i < getChildEdges().size(); i++) {
            auto& mklMemory = getChildEdgeAt(i)->getMemory();
            outputs.push_back(MKLDNNExtensionUtils::MKLMemoryToGenericMemory(mklMemory));
        }

        genericPrimitive->SetMemory(inputs, outputs);
        genericPrimitive->Execute();
    } else if (!impls.empty()) {
        execLayer();
    } else {
        THROW_IE_EXCEPTION << "Descriptor for generic primitive doesn't exist";
    }
}

bool MKLDNNGenericNode::created() const {
    return Generic == getType();
}

bool MKLDNNGenericNode::created(const MKLDNNExtensionManager::Ptr &extMgr) {
    if (getCnnLayer() && extMgr) {
        // We should save extension manager in otder to avoid situation when
        // it will destroyed before extensibility primitives
        extensionManager = extMgr;
        genericPrimitive.reset(extensionManager->CreateExtensionPrimitive(getCnnLayer()));
        extFactory.reset(extensionManager->CreateExtensionFactory(getCnnLayer()));

        if (genericPrimitive || extFactory)
            setType(Generic);
    }
    return created();
}

void MKLDNNGenericNode::cleanup() {
    MKLDNNNode::cleanup();
    extFactory.reset();
}

void MKLDNNGenericNode::execLayer() {
    bool isDynBatch = dynBatchLim > 0;
    std::vector<InferenceEngine::Blob::Ptr> inputs;
    std::vector<InferenceEngine::TensorDesc> inputDescs;
    std::vector<InferenceEngine::TensorDesc> outputDescs;
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        inputs.push_back(getParentEdgeAt(i)->getBlob());
        if (isDynBatch && dynBatchLim >= inputs[inputs.size() - 1]->getTensorDesc().getDims()[0]) {
            isDynBatch = false;
        } else {
            // TODO: Ask the right dims using getShape() from previous node
            inputDescs.push_back(inputs[inputs.size() - 1]->getTensorDesc());
            inputDescs[inputDescs.size() - 1].getDims()[0] = static_cast<size_t>(batchToProcess());
        }
    }

    if (isDynBatch) {
        auto sts = extFactory->getShapes(inputDescs, outputDescs, nullptr);
        if (sts != InferenceEngine::StatusCode::OK)
            isDynBatch = false;
    }

    if (isDynBatch) {
        for (size_t i = 0; i < inputs.size(); i++) {
            auto td = inputs[i]->getTensorDesc();
            td.setDims(inputDescs[i].getDims());
            inputs[i] = make_blob_with_precision(td, getParentEdgeAt(i)->getMemory().GetData());
        }
    }
    std::vector<InferenceEngine::Blob::Ptr> outputs;
    for (size_t i = 0; i < getChildEdges().size(); i++) {
        if (isDynBatch) {
            size_t idx = i >= outputDescs.size() ? 0 : i;
            auto td = getChildEdgeAt(i)->getBlob()->getTensorDesc();
            td.setDims(outputDescs[idx].getDims());
            outputs.push_back(make_blob_with_precision(td, getChildEdgeAt(i)->getMemory().GetData()));
        } else {
            outputs.push_back(getChildEdgeAt(i)->getBlob());
        }
    }
    auto * execImpl = dynamic_cast<InferenceEngine::ILayerExecImpl *>(impls[0].get());
    if (execImpl != nullptr) {
        InferenceEngine::ResponseDesc resp;
        InferenceEngine::StatusCode rc = execImpl->execute(inputs, outputs, &resp);
        if (rc != InferenceEngine::OK) {
            THROW_IE_EXCEPTION << resp.msg;
        }
    }
}

MKLDNNGenericNode::~MKLDNNGenericNode() {
    extFactory.reset();
    genericPrimitive.reset();
    extensionManager.reset();
}

void MKLDNNGenericNode::initDescriptor(const InferenceEngine::LayerConfig &config) {
    InferenceEngine::LayerConfig rightConfig = config;
    if (genericPrimitive) {
        for (auto &inConf : rightConfig.inConfs) {
            inConf.constant = false;
            inConf.inPlace = -1;
            if (inConf.desc.getLayout() == InferenceEngine::Layout::ANY) {
                inConf.desc = InferenceEngine::TensorDesc(inConf.desc.getPrecision(),
                                                          inConf.desc.getDims(),
                                                          InferenceEngine::TensorDesc::getLayoutByDims(
                                                                  inConf.desc.getDims()));
            } else {
                inConf.desc = InferenceEngine::TensorDesc(inConf.desc.getPrecision(),
                                                          inConf.desc.getDims(), {
                                                                  inConf.desc.getBlockingDesc().getBlockDims(),
                                                                  inConf.desc.getBlockingDesc().getOrder()
                                                          });
            }
        }
        for (auto &outConf : rightConfig.outConfs) {
            outConf.constant = false;
            outConf.inPlace = -1;
            if (outConf.desc.getLayout() == InferenceEngine::Layout::ANY) {
                outConf.desc = InferenceEngine::TensorDesc(outConf.desc.getPrecision(),
                                                           outConf.desc.getDims(),
                                                           InferenceEngine::TensorDesc::getLayoutByDims(
                                                                  outConf.desc.getDims()));
            } else {
                outConf.desc = InferenceEngine::TensorDesc(outConf.desc.getPrecision(),
                                                           outConf.desc.getDims(), {
                                                                   outConf.desc.getBlockingDesc().getBlockDims(),
                                                                   outConf.desc.getBlockingDesc().getOrder()
                                                           });
            }
        }
    } else {
        InferenceEngine::StatusCode rc;
        InferenceEngine::ResponseDesc resp;

        InferenceEngine::ILayerImpl::Ptr selectedImpl;
        for (size_t k = 0, t = 0; k < impls.size(); k++) {
            std::vector<InferenceEngine::LayerConfig> configs;
            rc = impls[k]->getSupportedConfigurations(configs, &resp);
            if (rc != InferenceEngine::OK) {
                THROW_IE_EXCEPTION << resp.msg;
            }
            for (size_t j = 0; j < configs.size(); j++, t++) {
                if (t == selectedPrimitiveDescriptorIndex) {
                    selectedImpl = impls[k];
                }
            }
        }

        for (size_t j = 0; j < rightConfig.inConfs.size(); j++) {
            if (getParentEdgeAt(j)->getParent()->getChildEdges().size() > 1) {
                rightConfig.inConfs[j].inPlace = -1;
            }
        }
        for (auto &outConf : rightConfig.outConfs) {
            if (outConf.inPlace < getParentEdges().size() &&
                getParentEdgeAt(static_cast<size_t>(outConf.inPlace))->getParent()->getChildEdges().size() > 1) {
                outConf.inPlace = -1;
            }
        }


        impls.clear();
        impls.emplace_back(selectedImpl);
        rc = impls[0]->init(rightConfig, &resp);
        if (rc != InferenceEngine::OK) {
            THROW_IE_EXCEPTION << resp.msg;
        }
    }

    getSelectedPrimitiveDescriptor()->getConfig() = rightConfig;
    bool isConst = !rightConfig.inConfs.empty() || !rightConfig.outConfs.empty();
    for (const auto &inConf : rightConfig.inConfs) {
        isConst = isConst && inConf.constant;
    }
    for (const auto &outConf : rightConfig.outConfs) {
        isConst = isConst && outConf.constant;
    }
    if (isConst) {
        constant = ConstantType::Const;
    }
}

void MKLDNNGenericNode::initOptimalPrimitiveDescriptor() {
    auto config = getSelectedPrimitiveDescriptor()->getConfig();
    if (genericPrimitive) {
        if (isInitConfig(config))
            return;

        for (size_t i = 0; i < config.inConfs.size(); i++) {
            if (!isUninitTensorDesc(config.inConfs[i].desc))
                continue;
            int num = getParentEdgeAt(i)->getInputNum();
            if (getParentEdgeAt(i)->getParent()->getSelectedPrimitiveDescriptor()->getConfig().outConfs.size() <= num)
                num = 0;
            auto parentConf = getParentEdgeAt(i)->getParent()->getSelectedPrimitiveDescriptor()->getConfig().outConfs[num];
            if (getParentEdgeAt(i)->getParent()->getSelectedPrimitiveDescriptor()) {
                if (num >= 0) {
                    if (isUninitTensorDesc(parentConf.desc) && (parentConf.inPlace >= 0 ||
                                                                parentConf.desc.getLayout() == InferenceEngine::Layout::ANY))
                        getParentEdgeAt(i)->getParent()->initOptimalPrimitiveDescriptor();
                    parentConf = getParentEdgeAt(i)->getParent()->getSelectedPrimitiveDescriptor()->getConfig().outConfs[num];
                    if (!isUninitTensorDesc(parentConf.desc) &&
                        MKLDNNExtensionUtils::initTensorsAreEqual(parentConf.desc, config.inConfs[i].desc)) {
                        if (config.inConfs[i].desc.getLayout() == InferenceEngine::Layout::ANY) {
                            for (size_t j = i + 1; j < config.inConfs.size(); j++) {
                                if (config.inConfs[j].desc.getLayout() == InferenceEngine::Layout::ANY) {
                                    config.inConfs[j].desc = parentConf.desc;
                                }
                            }
                            for (auto &outConf : config.outConfs) {
                                if (outConf.desc.getLayout() == InferenceEngine::Layout::ANY) {
                                    outConf.desc = parentConf.desc;
                                }
                            }
                        }
                        config.inConfs[i].desc = parentConf.desc;
                        continue;
                    }
                }
            }
            if (config.inConfs[i].desc.getLayout() != InferenceEngine::Layout::ANY) {
                config.inConfs[i].desc = InferenceEngine::TensorDesc(config.inConfs[i].desc.getPrecision(),
                                                                     config.inConfs[i].desc.getDims(), {
                                                                             config.inConfs[i].desc.getBlockingDesc().getBlockDims(),
                                                                             config.inConfs[i].desc.getBlockingDesc().getOrder()
                                                                     });
            } else if (parentConf.desc.getLayout() != InferenceEngine::Layout::ANY) {
                if (config.inConfs[i].desc.getLayout() == InferenceEngine::Layout::ANY) {
                    for (size_t j = i + 1; j < config.inConfs.size(); j++) {
                        if (config.inConfs[j].desc.getLayout() == InferenceEngine::Layout::ANY) {
                            config.inConfs[j].desc = InferenceEngine::TensorDesc(parentConf.desc.getPrecision(),
                                                                                 parentConf.desc.getDims(), {
                                                                                         parentConf.desc.getBlockingDesc().getBlockDims(),
                                                                                         parentConf.desc.getBlockingDesc().getOrder()
                                                                                 });
                        }
                    }
                    for (auto &outConf : config.outConfs) {
                        if (outConf.desc.getLayout() == InferenceEngine::Layout::ANY) {
                            outConf.desc = InferenceEngine::TensorDesc(parentConf.desc.getPrecision(),
                                                                       parentConf.desc.getDims(), {
                                                                               parentConf.desc.getBlockingDesc().getBlockDims(),
                                                                               parentConf.desc.getBlockingDesc().getOrder()
                                                                       });
                        }
                    }
                }
                config.inConfs[i].desc = InferenceEngine::TensorDesc(parentConf.desc.getPrecision(),
                                                                     parentConf.desc.getDims(), {
                                                                             parentConf.desc.getBlockingDesc().getBlockDims(),
                                                                             parentConf.desc.getBlockingDesc().getOrder()
                                                                     });
            } else {
                config.inConfs[i].desc = InferenceEngine::TensorDesc(config.inConfs[i].desc.getPrecision(),
                                                                     config.inConfs[i].desc.getDims(),
                                                                     InferenceEngine::TensorDesc::getLayoutByDims(config.inConfs[i].desc.getDims()));
            }
        }

        for (size_t i = 0; i < config.outConfs.size(); i++) {
            config.outConfs[i].desc = getConfiguredOutputDesc(config, i);
        }
    }

    initDescriptor(config);
}
