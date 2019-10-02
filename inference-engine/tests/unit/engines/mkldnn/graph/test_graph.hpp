// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mkldnn_plugin/nodes/list.hpp>
#include <mkldnn_plugin/mkldnn_graph.h>
#include <mkldnn_plugin/mkldnn_memory.h>
#include <mkldnn_plugin/mkldnn_extension_utils.h>
#include <mkldnn_plugin/mkldnn_graph_optimizer.h>
#include <mkldnn_plugin/nodes/mkldnn_input_node.h>
#include <functional>

#define GARB_VAL(x) ((x + 100.0f + sin(x)) / (x + 150.f))

class MKLDNNGraphTestClass: public MKLDNNPlugin::MKLDNNGraph {
private:
    MKLDNNPlugin::MKLDNNExtensionManager::Ptr extensionManager = std::make_shared<MKLDNNPlugin::MKLDNNExtensionManager>();

public:
    enum class CheckDynBatchType {
        Both,
        Parent,
        Child
    };
    MKLDNNGraphTestClass(): MKLDNNPlugin::MKLDNNGraph() {
        auto defaultExtensions = std::make_shared<InferenceEngine::Extensions::Cpu::MKLDNNExtensions<mkldnn::impl::cpu::cpu_isa_t::isa_any>>();
        extensionManager->AddExtension(defaultExtensions);

    }
    virtual ~MKLDNNGraphTestClass() = default;

    static std::string getStrPrimitiveDescriptorType(MKLDNNPlugin::impl_desc_type type) {
        std::string str_type;

        auto add_type = [&](std::string t) {
            if (!str_type.empty() && t.c_str()[0] != '_')
                str_type += "_";
            str_type += t;
        };

#define SEARCH_TYPE(_type)                                                                      \
    if ((type & MKLDNNPlugin::impl_desc_type::_type) == MKLDNNPlugin::impl_desc_type::_type)    \
        add_type(#_type)

        SEARCH_TYPE(undef);
        SEARCH_TYPE(reorder);
        SEARCH_TYPE(jit);
        SEARCH_TYPE(gemm);
        SEARCH_TYPE(ref);

        SEARCH_TYPE(avx512);
        SEARCH_TYPE(avx2);
        SEARCH_TYPE(sse42);
        SEARCH_TYPE(blas);
        SEARCH_TYPE(any);

        SEARCH_TYPE(winograd);
        SEARCH_TYPE(_dw);
        SEARCH_TYPE(_1x1);

        if (type == MKLDNNPlugin::impl_desc_type::unknown)
            str_type = "unknown";
        else if (str_type.empty())
            str_type = "undef";
        return str_type;
    }

    void PushInputData(const std::string& name, const InferenceEngine::Blob::Ptr &in, int batch) {
        if (!IsReady()) THROW_IE_EXCEPTION<< "Wrong state. Topology not ready.";

        auto input = inputNodes.find(name);
        if (input != inputNodes.end()) {
            MKLDNNPlugin::MKLDNNDims outDims;
            if(input->second->getChildEdgeAt(0)->getDims().ndims() == 0 )
                outDims = MKLDNNPlugin::MKLDNNDims(InferenceEngine::SizeVector(1,1));
            else
                outDims = input->second->getChildEdgeAt(0)->getDims();
            if (batch < 1)
                batch = outDims[0];

            const void *ext_data_ptr = in->cbuffer();
            void *inter_data_ptr = input->second->getChildEdgeAt(0)->getMemory().GetData();

            if (ext_data_ptr != inter_data_ptr)
                input->second->getChildEdgeAt(0)->getMemory().SetData(MKLDNNPlugin::MKLDNNExtensionUtils::IEPrecisionToDataType(in->getTensorDesc().getPrecision()),
                                                                      MKLDNNPlugin::MKLDNNMemory::GetPlainFormat(outDims), ext_data_ptr, in->byteSize() / outDims[0] * batch, false);

            // todo: make sure 'name' exists in this map...
            if (_meanImages.find(name) != _meanImages.end()) {
                if (in->getTensorDesc().getPrecision() == InferenceEngine::Precision::FP32) {
                    _meanImages[name].Subtract(outDims, reinterpret_cast<float *>(inter_data_ptr), in->getTensorDesc().getLayout());
                } else {
                    THROW_IE_EXCEPTION << "Mean image of type " << in->getTensorDesc().getPrecision().name() << " is unsupported";
                }
            }
        } else {
            THROW_IE_EXCEPTION << "Input blob for infer '" << name << "' doesn't correspond to input in network";
        }
    }

    void Infer(const InferenceEngine::BlobMap& inputs, InferenceEngine::BlobMap& result, int batch = -1) {
        try {
            // need to retain converted blobs until infer finish
            std::vector<InferenceEngine::Blob::Ptr> convertedInputs;
            for (auto input : inputs) {
                switch (input.second->getTensorDesc().getPrecision()) {
                    case InferenceEngine::Precision::FP32: {
                        InferenceEngine::TBlob<float> *in_f = nullptr;
                        in_f = dynamic_cast<InferenceEngine::TBlob<float> *>(input.second.get());
                        if (in_f == nullptr) {
                            FAIL() << "Input data precision not supported. Expected float.";
                        }

                        if (in_f->readOnly() == nullptr) {
                            THROW_IE_EXCEPTION << "Input data was not allocated.";
                        }
                    }
                    break;
                    case InferenceEngine::Precision::I32: {
                        InferenceEngine::TBlob<int32_t> *in_f = nullptr;
                        in_f = dynamic_cast<InferenceEngine::TBlob<int32_t> *>(input.second.get());
                        if (in_f == nullptr) {
                            FAIL() << "Input data precision not supported. Expected float.";
                        }

                        if (in_f->readOnly() == nullptr) {
                            THROW_IE_EXCEPTION << "Input data was not allocated.";
                        }
                    }
                    break;
                    case InferenceEngine::Precision::U16: {
                        InferenceEngine::TBlob<uint16_t> *in_f = nullptr;
                        in_f = dynamic_cast<InferenceEngine::TBlob<uint16_t> *>(input.second.get());
                        if (in_f == nullptr) {
                            FAIL() << "Input data precision not supported. Expected float.";
                        }

                        if (in_f->readOnly() == nullptr) {
                            THROW_IE_EXCEPTION << "Input data was not allocated.";
                        }
                    }
                    break;
                    case InferenceEngine::Precision::I16: {
                        InferenceEngine::TBlob<int16_t> *in_f = nullptr;
                        in_f = dynamic_cast<InferenceEngine::TBlob<int16_t> *>(input.second.get());
                        if (in_f == nullptr) {
                            FAIL() << "Input data precision not supported. Expected float.";
                        }

                        if (in_f->readOnly() == nullptr) {
                            THROW_IE_EXCEPTION << "Input data was not allocated.";
                        }
                    }
                    break;
                    case InferenceEngine::Precision::U8: {
                        InferenceEngine::TBlob<uint8_t> *in_f = nullptr;
                        in_f = dynamic_cast<InferenceEngine::TBlob<uint8_t> *>(input.second.get());
                        if (in_f == nullptr) {
                            FAIL() << "Input data precision not supported. Expected float.";
                        }

                        if (in_f->readOnly() == nullptr) {
                            THROW_IE_EXCEPTION << "Input data was not allocated.";
                        }
                    }
                    break;
                    case InferenceEngine::Precision::I8: {
                        InferenceEngine::TBlob<int8_t> *in_f = nullptr;
                        in_f = dynamic_cast<InferenceEngine::TBlob<int8_t> *>(input.second.get());
                        if (in_f == nullptr) {
                            FAIL() << "Input data precision not supported. Expected float.";
                        }

                        if (in_f->readOnly() == nullptr) {
                            THROW_IE_EXCEPTION << "Input data was not allocated.";
                        }
                    }
                    break;
                    default:
                        THROW_IE_EXCEPTION << "Unsupported input precision " << input.second->getTensorDesc().getPrecision();
                }

                PushInputData(input.first, input.second, batch);
            }
            MKLDNNPlugin::MKLDNNGraph::Infer(batch);
        } catch (const std::exception &e) {
            FAIL() << e.what();
        }

        PullOutputData(result);
    }

    std::vector<MKLDNNPlugin::MKLDNNNodePtr>& getNodes() {
        return graphNodes;
    }

    void CreateGraph(InferenceEngine::ICNNNetwork &network, const MKLDNNPlugin::MKLDNNExtensionManager::Ptr& extMgr) {
        MKLDNNGraph::CreateGraph(network, extMgr);
    }

    void CreateGraph(InferenceEngine::ICNNNetwork &network) {
        CreateGraph(network, extensionManager);
    }

    void checkDynBatch(InferenceEngine::BlobMap& srcs, InferenceEngine::BlobMap& outputBlobs, int batch, size_t MB,
                       const std::function<bool (const MKLDNNPlugin::MKLDNNNodePtr&)>& comp, CheckDynBatchType type = CheckDynBatchType::Both) {
        for (auto &node : getNodes()) {
            if (comp(node)) {
                auto inputBlob = node->getParentEdgeAt(0)->getBlob();
                auto *data = inputBlob->buffer().as<float *>();
                size_t dataSize = inputBlob->getTensorDesc().getBlockingDesc().getStrides()[0] * MB;
                for (size_t j = 0; j < dataSize; j++) {
                    data[j] = GARB_VAL(j);
                }

                auto outputBlob = node->getChildEdgeAt(0)->getBlob();
                data = outputBlob->buffer().as<float *>();
                dataSize = outputBlob->getTensorDesc().getBlockingDesc().getStrides()[0] * MB;
                for (size_t j = 0; j < dataSize; j++) {
                    data[j] = GARB_VAL(j);
                }
            }
        }

        Infer(srcs, outputBlobs, batch);

        for (auto &node : getNodes()) {
            if (comp(node)) {
                auto inputBlob = node->getParentEdgeAt(0)->getBlob();
                auto *data = inputBlob->buffer().as<float *>();
                auto inputNoBatchSize = inputBlob->getTensorDesc().getBlockingDesc().getStrides()[0];
                for (size_t i = 0; i < batch; i++) {
                    for (size_t j = 0; j < inputNoBatchSize; j++) {
                        ASSERT_NE(data[i*inputNoBatchSize + j], GARB_VAL(i*inputNoBatchSize + j));
                    }
                }

                if (type == CheckDynBatchType::Both || type == CheckDynBatchType::Parent) {
                    for (size_t i = static_cast<size_t>(batch); i < MB; i++) {
                        for (size_t j = 0; j < inputNoBatchSize; j++) {
                            ASSERT_NEAR(data[i * inputNoBatchSize + j],
                                        GARB_VAL(i * inputNoBatchSize + j), 0.001f);
                        }
                    }
                }

                auto outputBlob = node->getChildEdgeAt(0)->getBlob();
                data = outputBlob->buffer().as<float *>();
                auto outputNoBatchSize = outputBlob->getTensorDesc().getBlockingDesc().getStrides()[0];
                for (size_t i = 0; i < batch; i++) {
                    for (size_t j = 0; j < outputNoBatchSize; j++) {
                        ASSERT_NE(data[i*outputNoBatchSize + j], GARB_VAL(i*outputNoBatchSize + j));
                    }
                }
                if (type == CheckDynBatchType::Both || type == CheckDynBatchType::Child) {
                    for (size_t i = static_cast<size_t>(batch); i < MB; i++) {
                        for (size_t j = 0; j < outputNoBatchSize; j++) {
                            ASSERT_NEAR(data[i * outputNoBatchSize + j],
                                        GARB_VAL(i * outputNoBatchSize + j), 0.001f);
                        }
                    }
                }
            }
        }
    }
};
