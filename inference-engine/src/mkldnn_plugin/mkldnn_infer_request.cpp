// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_infer_request.h"
#include "mkldnn_extension_utils.h"
#include <vector>
#include <string>
#include <map>
#include <blob_factory.hpp>
#include <nodes/mkldnn_concat_node.h>
#include <nodes/mkldnn_split_node.h>
#include <ie_compound_blob.h>
#include "inference_engine.hpp"
#include "mkldnn_exec_network.h"
#include "mkldnn_itt.h"

MKLDNNPlugin::MKLDNNInferRequest::MKLDNNInferRequest(InferenceEngine::InputsDataMap     networkInputs,
                                                     InferenceEngine::OutputsDataMap    networkOutputs,
                                                     MKLDNNExecNetwork::Ptr             execNetwork_)
: InferRequestInternal(networkInputs, networkOutputs)
, execNetwork(execNetwork_) {
    auto id = (execNetwork->_numRequests)++;
    profilingTask = openvino::itt::handle("MKLDNN_INFER_" + execNetwork->_name + "_" + std::to_string(id));

    if (execNetwork->_graphs.size() == 0)
        THROW_IE_EXCEPTION << "No graph was found";
    graph = execNetwork->_graphs.begin()->get();
    for (const auto& it : _networkInputs) {
        InferenceEngine::Blob::Ptr blob;
        MKLDNNInferRequest::GetBlob(it.first.c_str(), blob);
    }
    // Allocate all output blobs
    for (const auto& it : _networkOutputs) {
        InferenceEngine::Blob::Ptr blob;
        MKLDNNInferRequest::GetBlob(it.first.c_str(), blob);
    }
}

MKLDNNPlugin::MKLDNNInferRequest::~MKLDNNInferRequest() {
    --(execNetwork->_numRequests);
}

template <typename T>
void MKLDNNPlugin::MKLDNNInferRequest::pushInput(const std::string& inputName, InferenceEngine::Blob::Ptr& inputBlob) {
    InferenceEngine::TBlob<T> *in_f = dynamic_cast<InferenceEngine::TBlob<T> *>(inputBlob.get());

    if (in_f == nullptr) {
        THROW_IE_EXCEPTION << "Input data precision not supported. Expected float.";
    }

    if (in_f->readOnly() == nullptr) {
        THROW_IE_EXCEPTION << "Input data was not allocated.";
    }

    graph->PushInputData(inputName, inputBlob);
}

namespace {

template <typename T>
void copyToFloat(float* dst, const InferenceEngine::Blob* src) {
    if (!dst) {
        return;
    }
    const InferenceEngine::TBlob<T>* t_blob = dynamic_cast<const InferenceEngine::TBlob<T>*>(src);
    if (t_blob == nullptr) {
        THROW_IE_EXCEPTION << "input type is " << src->getTensorDesc().getPrecision() << " but input is not "
                           << typeid(T).name();
    }

    const T* srcPtr = t_blob->readOnly();
    if (srcPtr == nullptr) {
        THROW_IE_EXCEPTION << "Input data was not allocated.";
    }
    for (size_t i = 0; i < t_blob->size(); i++) dst[i] = srcPtr[i];
}

}  // namespace

void MKLDNNPlugin::MKLDNNInferRequest::InferImpl() {
    using namespace openvino::itt;
    OV_ITT_SCOPED_TASK(itt::domains::MKLDNNPlugin, profilingTask);

    graph = execNetwork->_graphs.local().get();
    {
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

            InferenceEngine::Blob::Ptr iconv;
            InferenceEngine::TBlob<float> *in_f = nullptr;
            switch (input.second->getTensorDesc().getPrecision()) {
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
                    iconv = InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32,
                                                                        input.second->getTensorDesc().getDims(),
                                                                        input.second->getTensorDesc().getLayout()});
                    convertedInputs.push_back(iconv);
                    iconv->allocate();
                    in_f = dynamic_cast<InferenceEngine::TBlob<float> *>(iconv.get());
                    if (in_f == nullptr)
                        THROW_IE_EXCEPTION << "Cannot get TBlob";
                    copyToFloat<uint16_t>(in_f->data(), input.second.get());
                    pushInput<float>(input.first, iconv);
                    break;
                case InferenceEngine::Precision::I16:
                    if (graph->hasMeanImageFor(input.first)) {
                        // If a mean image exists, we convert the blob and send FP32
                        iconv = InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32,
                                                                            input.second->getTensorDesc().getDims(),
                                                                            input.second->getTensorDesc().getLayout()});
                        convertedInputs.push_back(iconv);
                        iconv->allocate();
                        in_f = dynamic_cast<InferenceEngine::TBlob<float> *>(iconv.get());
                        if (in_f == nullptr)
                            THROW_IE_EXCEPTION << "Cannot get TBlob";
                        copyToFloat<int16_t>(in_f->data(), input.second.get());
                        pushInput<float>(input.first, iconv);
                    } else {
                        // Instead we can send I16 directly
                        pushInput<int16_t>(input.first, input.second);
                    }
                    break;
                case InferenceEngine::Precision::U8:
                case InferenceEngine::Precision::BOOL:
                    if (graph->hasMeanImageFor(input.first)) {
                        // If a mean image exists, we convert the blob and send FP32
                        iconv = InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32,
                                                                            input.second->getTensorDesc().getDims(),
                                                                            input.second->getTensorDesc().getLayout()});
                        convertedInputs.push_back(iconv);
                        iconv->allocate();
                        in_f = dynamic_cast<InferenceEngine::TBlob<float> *>(iconv.get());
                        if (in_f == nullptr)
                            THROW_IE_EXCEPTION << "Cannot get TBlob";
                        copyToFloat<uint8_t>(in_f->data(), input.second.get());
                        pushInput<float>(input.first, iconv);
                    } else {
                        // Instead we can send I8 directly
                        pushInput<uint8_t>(input.first, input.second);
                    }
                    break;
                default:
                    THROW_IE_EXCEPTION << "Unsupported input precision " << input.second->getTensorDesc().getPrecision();
            }
        }
    }

    graph->Infer(m_curBatch);

    graph->PullOutputData(_outputs);
}

void MKLDNNPlugin::MKLDNNInferRequest::GetPerformanceCounts(
        std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> &perfMap) const {
    if (!graph || !graph->IsReady())
        THROW_IE_EXCEPTION << "Graph is not ready!";
    graph->GetPerfData(perfMap);
}

void MKLDNNPlugin::MKLDNNInferRequest::GetBlob(const char *name, InferenceEngine::Blob::Ptr &data) {
    OV_ITT_SCOPED_TASK(itt::domains::MKLDNNPlugin, "GetBlob");

    if (!graph || !graph->IsReady())
        THROW_IE_EXCEPTION << "Graph is not ready!";

    InferenceEngine::BlobMap blobs;
    graph->getInputBlobs(blobs);

    if (blobs.find(name) != blobs.end()) {
        // ROI blob is returned only if it was set previously.
        auto it = _preProcData.find(name);
        if (it != _preProcData.end()) {
            data = it->second->getRoiBlob();
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
    OV_ITT_SCOPED_TASK(itt::domains::MKLDNNPlugin, "SetBlob");
    if (name == nullptr) {
        THROW_IE_EXCEPTION << NOT_FOUND_str + "Failed to set blob with empty name";
    }

    if (!data)
        THROW_IE_EXCEPTION << NOT_ALLOCATED_str << "Failed to set empty blob with name: \'" << name << "\'";
    const bool compoundBlobPassed = data->is<InferenceEngine::CompoundBlob>();
    if (!compoundBlobPassed && data->buffer() == nullptr)
        THROW_IE_EXCEPTION << "Input data was not allocated. Input name: \'" << name << "\'";
    if (data->size() == 0) {
        THROW_IE_EXCEPTION << "Input data is empty. Input name: \'" << name << "\'";
    }

    InferenceEngine::InputInfo::Ptr foundInput;
    InferenceEngine::DataPtr foundOutput;
    size_t dataSize = data->size();
    if (findInputAndOutputBlobByName(name, foundInput, foundOutput)) {
        if (foundInput->getPrecision() != data->getTensorDesc().getPrecision()) {
            THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str << "Failed to set Blob with precision "
                               << data->getTensorDesc().getPrecision();
        }

        const bool preProcRequired = preProcessingRequired(foundInput, data);
        if (compoundBlobPassed && !preProcRequired) {
            THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str
                               << "cannot set compound blob: supported only for input pre-processing";
        }

        if (preProcRequired) {
            if (_preProcData.find(name) == _preProcData.end()) {
                _preProcData.emplace(name, InferenceEngine::CreatePreprocDataHelper());
            }
            _preProcData[name]->isApplicable(data, _inputs[name]);
            // Stores the given blob as ROI blob. It will be used to fill in network input during
            // pre-processing
            _preProcData[name]->setRoiBlob(data);
        } else {
            size_t inputSize = foundInput->getTensorDesc().getLayout() != InferenceEngine::Layout::SCALAR
                ? InferenceEngine::details::product(foundInput->getTensorDesc().getDims())
                : 1;
            if (dataSize != inputSize) {
                THROW_IE_EXCEPTION << "Input blob size is not equal network input size ("
                                   << dataSize << "!=" << inputSize << ").";
            }

            if (foundInput->getTensorDesc().getDims() != data->getTensorDesc().getDims()) {
                THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str << "Failed to set input Blob. Dimensions mismatch.";
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
        if (compoundBlobPassed) {
            THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str
                               << "cannot set compound blob: supported only for input pre-processing";
        }
        size_t outputSize = foundOutput->getTensorDesc().getLayout() != InferenceEngine::Layout::SCALAR
            ? InferenceEngine::details::product(foundOutput->getDims())
            : 1;
        if (dataSize != outputSize) {
            THROW_IE_EXCEPTION << "Output blob size is not equal network output size ("
                               << dataSize << "!=" << outputSize << ").";
        }
        if (foundOutput->getTensorDesc().getDims() != data->getTensorDesc().getDims()) {
            THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str << "Failed to set output Blob. Dimensions mismatch.";
        }
        if (foundOutput->getPrecision() != data->getTensorDesc().getPrecision()) {
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

static inline void changeEdgePtr(const MKLDNNPlugin::MKLDNNEdgePtr &edge, void *newPtr) {
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
#if defined(COMPILED_CPU_MKLDNN_CONCAT_NODE)
                auto* concat = dynamic_cast<MKLDNNConcatNode *>(child.get());
                if (canBeInPlace && concat && concat->isOptimized())
                    canBeInPlace = false;
#endif
                // Cannot be in-place before split because split is using different ptrs without offsets
#if defined(COMPILED_CPU_MKLDNN_SPLIT_NODE)
                auto* split = dynamic_cast<MKLDNNSplitNode *>(child.get());
                if (canBeInPlace && split)
                    canBeInPlace = false;
#endif

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


void MKLDNNPlugin::MKLDNNInferRequest::SetBatch(int new_batch) {
    if (!graph->getProperty().enableDynamicBatch)
        THROW_IE_EXCEPTION << "Dynamic batch is not enabled.";

    if (new_batch < 1 || new_batch > graph->getProperty().batchLimit) {
        THROW_IE_EXCEPTION << "Invalid dynamic batch size " << new_batch <<
            " for this request.";
    }

    m_curBatch = new_batch;
}
