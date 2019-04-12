// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_infer_request.h"
#include "mkldnn_extension_utils.h"
#include "mkldnn_streams.h"
#include <vector>
#include <string>
#include <map>
#include <blob_factory.hpp>
#include <nodes/mkldnn_concat_node.h>
#include <nodes/mkldnn_split_node.h>

MKLDNNPlugin::MKLDNNInferRequest::MKLDNNInferRequest(InferenceEngine::InputsDataMap networkInputs,
                                                     InferenceEngine::OutputsDataMap networkOutputs)
        : InferRequestInternal(networkInputs, networkOutputs) {}


template <typename T> void MKLDNNPlugin::MKLDNNInferRequest::pushInput(const std::string& inputName, InferenceEngine::Blob::Ptr& inputBlob) {
    InferenceEngine::TBlob<T> *in_f = dynamic_cast<InferenceEngine::TBlob<T> *>(inputBlob.get());

    if (in_f == nullptr) {
        THROW_IE_EXCEPTION << "Input data precision not supported. Expected float.";
    }

    if (in_f->readOnly() == nullptr) {
        THROW_IE_EXCEPTION << "Input data was not allocated.";
    }

    graph->PushInputData(inputName, inputBlob);
}

void MKLDNNPlugin::MKLDNNInferRequest::InferImpl() {
    IE_PROFILING_AUTO_SCOPE(MKLDNN_INFER)
    if (!graph || !graph->IsReady()) {
        THROW_IE_EXCEPTION << "Network not loaded.";
    }
    auto infer = [this] {
        // execute input pre-processing.
        execDataPreprocessing(_inputs);

        changeDefaultPtr();
        // need to retain converted blobs until infer finish
        std::vector<InferenceEngine::Blob::Ptr> convertedInputs;
        for (auto input : _inputs) {
            if (!_networkInputs[input.first]) {
                THROW_IE_EXCEPTION <<
                                   "input blobs map contains not registered during IInferencePlugin::LoadNetwork blob with name "
                                   << input.first;
            }
            /*if (_networkInputs[input.first]->getInputPrecision() != input.second->precision()) {
                THROW_IE_EXCEPTION << "Different input precision for input " << input.first
                                   << " registered in IInferencePlugin::LoadNetwork network and IInferencePlugin::Infer. "
                                   << _networkInputs[input.first]->getInputPrecision() << " vs "
                                   << input.second->precision();
            }*/



            InferenceEngine::Blob::Ptr iconv;
            InferenceEngine::TBlob<float> *in_f = nullptr;
            switch (input.second->precision()) {
                case InferenceEngine::Precision::FP32:
                    pushInput<float>(input.first, input.second);
                    break;
                case InferenceEngine::Precision::I32:
                    pushInput<int32_t>(input.first, input.second);
                    break;
                case InferenceEngine::Precision::I8:
                    pushInput<int8_t>(input.first, input.second);
                    break;
                case InferenceEngine::Precision::U16:
                    // U16 is unsupported by mkldnn, so here we convert the blob and send FP32
                    iconv = InferenceEngine::make_shared_blob<float, const InferenceEngine::SizeVector>(
                            InferenceEngine::Precision::FP32,
                            input.second->getTensorDesc().getLayout(), input.second->dims());
                    convertedInputs.push_back(iconv);
                    iconv->allocate();
                    in_f = dynamic_cast<InferenceEngine::TBlob<float> *>(iconv.get());
                    InferenceEngine::copyToFloat<uint16_t>(in_f->data(), input.second.get());
                    pushInput<float>(input.first, iconv);
                    break;
                case InferenceEngine::Precision::I16:
                    if (graph->hasMeanImageFor(input.first)) {
                        // If a mean image exists, we convert the blob and send FP32
                        iconv = InferenceEngine::make_shared_blob<float, const InferenceEngine::SizeVector>(
                                InferenceEngine::Precision::FP32,
                                input.second->getTensorDesc().getLayout(), input.second->dims());
                        convertedInputs.push_back(iconv);
                        iconv->allocate();
                        in_f = dynamic_cast<InferenceEngine::TBlob<float> *>(iconv.get());
                        InferenceEngine::copyToFloat<int16_t>(in_f->data(), input.second.get());
                        pushInput<float>(input.first, iconv);
                    } else {
                        // Instead we can send I16 directly
                        pushInput<int16_t>(input.first, input.second);
                    }
                    break;
                case InferenceEngine::Precision::U8:
                    if (graph->hasMeanImageFor(input.first)) {
                        // If a mean image exists, we convert the blob and send FP32
                        iconv = InferenceEngine::make_shared_blob<float, const InferenceEngine::SizeVector>(
                                InferenceEngine::Precision::FP32,
                                input.second->getTensorDesc().getLayout(), input.second->dims());
                        convertedInputs.push_back(iconv);
                        iconv->allocate();
                        in_f = dynamic_cast<InferenceEngine::TBlob<float> *>(iconv.get());
                        InferenceEngine::copyToFloat<uint8_t>(in_f->data(), input.second.get());
                        pushInput<float>(input.first, iconv);
                    } else {
                        // Instead we can send I8 directly
                        pushInput<uint8_t>(input.first, input.second);
                    }
                    break;
                default:
                    THROW_IE_EXCEPTION << "Unsupported input precision " << input.second->precision();
            }
        }
        graph->Infer(m_curBatch);
        graph->PullOutputData(_outputs);
    };
#if IE_THREAD == IE_THREAD_TBB
    auto_scope_observing observer(graph->ptrObserver);
    // a TBB arena is made "this" for Infer call via executing lambda for the arena
    graph->ptrArena->execute([&] { infer(); });
#else
    infer();
#endif
}

void MKLDNNPlugin::MKLDNNInferRequest::GetPerformanceCounts(
        std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> &perfMap) const {
    if (!graph || !graph->IsReady())
        THROW_IE_EXCEPTION << "Graph is not ready!";
    graph->GetPerfData(perfMap);
}

void MKLDNNPlugin::MKLDNNInferRequest::GetBlob(const char *name, InferenceEngine::Blob::Ptr &data) {
    if (!graph || !graph->IsReady())
        THROW_IE_EXCEPTION << "Graph is not ready!";

    InferenceEngine::BlobMap blobs;
    graph->getInputBlobs(blobs);

    if (blobs.find(name) != blobs.end()) {
        // ROI blob is returned only if it was set previously.
        auto it = _preProcData.find(name);
        if (it != _preProcData.end()) {
            data = it->second.getRoiBlob();
            return;
        }

        if (_inputs.find(name) != _inputs.end()) {
            data = _inputs[name];
            checkBlob(data, name, true);
            return;
        }

        InferenceEngine::TensorDesc desc = blobs[name]->getTensorDesc();
        InferenceEngine::Precision originPrecision = blobs[name]->getTensorDesc().getPrecision();
        if (_networkInputs.find(name) != _networkInputs.end()) {
            InferenceEngine::Layout l = _networkInputs[name]->getLayout();
            InferenceEngine::Precision p = _networkInputs[name]->getPrecision();
            InferenceEngine::SizeVector dims = _networkInputs[name]->getTensorDesc().getDims();

            desc = InferenceEngine::TensorDesc(p, dims, l);
        }

        _inputs[name] = make_blob_with_precision(desc);
        _inputs[name]->allocate();
        if (desc.getPrecision() == originPrecision &&
                graph->_meanImages.find(name) == graph->_meanImages.end() && !graph->getProperty().batchLimit) {
            externalPtr[name] = _inputs[name]->buffer();
        }
        data = _inputs[name];
        checkBlob(data, name, true);
        return;
    }
    blobs.clear();
    graph->getOutputBlobs(blobs);

    if (blobs.find(name) != blobs.end()) {
        if (_outputs.find(name) != _outputs.end()) {
            data = _outputs[name];
            checkBlob(data, name, false);
            return;
        }

        _outputs[name] = make_blob_with_precision(blobs[name]->getTensorDesc());
        _outputs[name]->allocate();
        if (blobs[name]->getTensorDesc().getPrecision() == InferenceEngine::Precision::FP32 &&
                !graph->getProperty().batchLimit) {
            externalPtr[name] = _outputs[name]->buffer();
        }
        data = _outputs[name];
        checkBlob(data, name, false);
        return;
    }
    THROW_IE_EXCEPTION << "Cannot find blob with name: " << name;
}

void MKLDNNPlugin::MKLDNNInferRequest::SetBlob(const char *name, const InferenceEngine::Blob::Ptr &data) {
    if (!data)
        THROW_IE_EXCEPTION << NOT_ALLOCATED_str << "Failed to set empty blob with name: \'" << name << "\'";
    if (data->buffer() == nullptr)
        THROW_IE_EXCEPTION << "Input data was not allocated. Input name: \'" << name << "\'";
    if (name == nullptr) {
        THROW_IE_EXCEPTION << NOT_FOUND_str + "Failed to set blob with empty name";
    }
    InferenceEngine::InputInfo::Ptr foundInput;
    InferenceEngine::DataPtr foundOutput;
    size_t dataSize = data->size();
    if (findInputAndOutputBlobByName(name, foundInput, foundOutput)) {
        if (foundInput->getInputPrecision() != data->precision()) {
            THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str << "Failed to set Blob with precision "
                               << data->precision();
        }

        if (foundInput->getPreProcess().getResizeAlgorithm() != InferenceEngine::ResizeAlgorithm::NO_RESIZE) {
            PreProcessData::isApplicable(data, _inputs[name]);
            // Stores the given blob as ROI blob. It will be used to fill in network input during pre-processing.
            _preProcData[name].setRoiBlob(data);
        } else {
            size_t inputSize = InferenceEngine::details::product(foundInput->getDims());
            if (dataSize != inputSize) {
                THROW_IE_EXCEPTION << "Input blob size is not equal network input size ("
                                   << dataSize << "!=" << inputSize << ").";
            }

            if (data->getTensorDesc().getPrecision() == InferenceEngine::Precision::FP32 &&
                graph->_meanImages.find(name) == graph->_meanImages.end() && !graph->getProperty().batchLimit) {
                externalPtr[name] = data->buffer();
            } else if (externalPtr.find(name) != externalPtr.end()) {
                externalPtr.erase(name);
            }
            _inputs[name] = data;
        }
    } else {
        size_t outputSize = InferenceEngine::details::product(foundOutput->getDims());
        if (dataSize != outputSize) {
            THROW_IE_EXCEPTION << "Output blob size is not equal network output size ("
                               << dataSize << "!=" << outputSize << ").";
        }
        if (foundOutput->getPrecision() != data->precision()) {
            THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str
                               << "Failed to set Blob with precision not corresponding to user output precision";
        }
        if (data->getTensorDesc().getPrecision() == InferenceEngine::Precision::FP32 &&
                !graph->getProperty().batchLimit) {
            externalPtr[name] = data->buffer();
        } else if (externalPtr.find(name) != externalPtr.end()) {
            externalPtr.erase(name);
        }
        _outputs[name] = data;
    }
}

static inline void changeEdgePtr(MKLDNNPlugin::MKLDNNEdgePtr edge, void *newPtr) {
    edge->getMemory().GetPrimitivePtr()->set_data_handle(newPtr);
}

void MKLDNNPlugin::MKLDNNInferRequest::changeDefaultPtr() {
    for (auto& it : externalPtr) {
        auto input = graph->inputNodes.find(it.first);
        if (input != graph->inputNodes.end()) {
            if (input->second->getChildEdgeAt(0)->getMemory().GetPrimitive().get_data_handle() == it.second)
                continue;
            // Input cannot be in-place with other primitives
            bool canBeInPlace = true;
            for (size_t i = 0; canBeInPlace && i < input->second->getChildEdges().size(); i++) {
                auto& child = input->second->getChildEdgeAt(i)->getChild();
                if (child->isConstant())
                    canBeInPlace = false;
                auto* concat = dynamic_cast<MKLDNNConcatNode *>(child.get());
                if (canBeInPlace && concat && concat->isOptimized())
                    canBeInPlace = false;
                // Cannot be in-place before split because split is using different ptrs without offsets
                auto* split = dynamic_cast<MKLDNNSplitNode *>(child.get());
                if (canBeInPlace && split)
                    canBeInPlace = false;

                if (child->isInplace())
                    canBeInPlace = false;
                for (size_t j = 0; canBeInPlace && j < child->getChildEdges().size(); j++) {
                    if (child->getChildEdgeAt(j)->getMemory().GetPrimitive().get_data_handle() ==
                            input->second->getChildEdgeAt(i)->getMemory().GetPrimitive().get_data_handle())
                        canBeInPlace = false;
                }
            }
            for (size_t i = 0; canBeInPlace && i < input->second->getChildEdges().size(); i++) {
                changeEdgePtr(input->second->getChildEdgeAt(i), it.second);
            }
            continue;
        }

        MKLDNNNodePtr output;
        for (auto& out : graph->outputNodes) {
            if (out->getName() == "out_" + it.first) {
                output = out;
                break;
            }
        }
        if (output) {
            if (output->getParentEdgeAt(0)->getMemory().GetPrimitive().get_data_handle() == it.second)
                continue;
            bool canBeInPlace = true;
            void * defaultPtr = output->getParentEdgeAt(0)->getMemory().GetPrimitivePtr()->get_data_handle();
            // Cannot be in-place after concat because concat is using different ptrs without offsets
            auto parent = output->getParentEdgeAt(0)->getParent();
            MKLDNNNodePtr previousParent;
            do {
                previousParent = parent;
                if (parent->getChildEdges().size() != 1 || parent->isConstant() || parent->isInplace()) {
                    canBeInPlace = false;
                    break;
                }

                for (size_t i = 0; i < parent->getParentEdges().size(); i++) {
                    if (parent->getParentEdgeAt(i)->getMemory().GetPrimitivePtr()->get_data_handle() == defaultPtr) {
                        parent = parent->getParentEdgeAt(i)->getParent();
                        break;
                    }
                }
            } while (previousParent != parent);
            if (canBeInPlace)
                changeEdgePtr(output->getParentEdgeAt(0), it.second);
            continue;
        }
        THROW_IE_EXCEPTION << "Cannot find input/output blob: " << it.first;
    }
}

void MKLDNNPlugin::MKLDNNInferRequest::SetGraph(const MKLDNNPlugin::MKLDNNGraph::Ptr &graph) {
    this->graph = graph;

    InferenceEngine::BlobMap blobs;
    this->graph->getInputBlobs(blobs);
    for (const auto& it : blobs) {
        InferenceEngine::Blob::Ptr blob;
        GetBlob(it.first.c_str(), blob);
    }
    blobs.clear();
    this->graph->getOutputBlobs(blobs);
    for (const auto& it : blobs) {
        InferenceEngine::Blob::Ptr blob;
        GetBlob(it.first.c_str(), blob);
    }
}

void MKLDNNPlugin::MKLDNNInferRequest::SetBatch(int new_batch) {
    if (!graph->getProperty().enableDynamicBatch)
        THROW_IE_EXCEPTION << "Dynamic batch is not enabled.";

    if (new_batch < 1 || new_batch > graph->getProperty().batchLimit) {
        THROW_IE_EXCEPTION << "Invalid dynamic batch size " << new_batch <<
            " for this request.";
    }

    m_curBatch = new_batch;
}
