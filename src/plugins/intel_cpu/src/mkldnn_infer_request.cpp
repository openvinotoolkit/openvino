// Copyright (C) 2018-2022 Intel Corporation
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
#include <ie_common.h>
#include "mkldnn_exec_network.h"
#include "mkldnn_itt.h"
#include "nodes/common/cpu_convert.h"
#include "mkldnn_memory_state.h"
#include "nodes/mkldnn_memory_node.hpp"
#include "nodes/common/cpu_memcpy.h"
#include "mkldnn_async_infer_request.h"
#include <debug.h>
#include "utils/general_utils.h"
#include "utils/cpu_utils.hpp"
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include <transformations/utils/utils.hpp>
#include <ie_ngraph_utils.hpp>

void MKLDNNPlugin::MKLDNNInferRequestBase::CreateInferRequest() {
    auto id = (execNetwork->_numRequests)++;
    profilingTask = openvino::itt::handle("MKLDNN_INFER_" + execNetwork->_name + "_" + std::to_string(id));

    if (execNetwork->_graphs.size() == 0)
        IE_THROW() << "No graph was found";
    graph = &(execNetwork->GetGraph()._graph);

    initBlobs();

    // Save all MemoryLayer data tensors. Will use insight about mechanics
    // of MemoryLayer implementation. It uses output edge of MemoryLayer
    // producer as storage for tensor to keep it between infer calls.
    for (auto& node : graph->GetNodes()) {
        if (node->getType() == MemoryInput) {
            auto memoryNode = dynamic_cast<MKLDNNMemoryInputNode*>(node.get());
            if (!memoryNode) {
                IE_THROW() << "Cannot cast " << node->getName() << " to MKLDNNMemoryInputNode";
            }
            auto state_store = memoryNode->getStore();
            auto state_name = memoryNode->getId();

            // Remove suffix with pair ID. Internal information.
            auto suffix_idx = state_name.find("/id=");
            if (suffix_idx != std::string::npos)
                state_name = state_name.substr(0, suffix_idx);

            memoryStates.emplace_back(new MKLDNNVariableState(state_name, state_store));
        }
    }
}

MKLDNNPlugin::MKLDNNInferRequestBase::~MKLDNNInferRequestBase() {
    --(execNetwork->_numRequests);
}

void MKLDNNPlugin::MKLDNNInferRequestBase::pushInput(const std::string& inputName, InferenceEngine::Blob::Ptr& inputBlob, InferenceEngine::Precision inPrec) {
    auto& tensorDesc = inputBlob->getTensorDesc();
    bool needConvert = inPrec != tensorDesc.getPrecision();

    const void* srcData = inputBlob->cbuffer().as<const void *>();
    if (srcData == nullptr) {
        IE_THROW() << "Input blob has no allocated memory";
    }

    InferenceEngine::Blob::Ptr iconv;
    if (needConvert) {
        iconv = make_blob_with_precision(inPrec, InferenceEngine::TensorDesc(inPrec, tensorDesc.getDims(), tensorDesc.getLayout()));
        iconv->allocate();
        if (inputBlob->size() != iconv->size())
            IE_THROW() << "Can't copy tensor: input and converted tensors have different number of elements: " << inputBlob->size() << " and "
                               << iconv->size();

        void *dstData = iconv->buffer().as<void *>();
        if (dstData == nullptr) {
            IE_THROW() << "Converted input blob has no allocated memory";
        }
        cpu_convert(srcData, dstData, tensorDesc.getPrecision(), iconv->getTensorDesc().getPrecision(), iconv->size());
    }

    graph->PushInputData(inputName, needConvert ? iconv : inputBlob);
}

void MKLDNNPlugin::MKLDNNInferRequestBase::PushStates() {
    for (auto &node : graph->GetNodes()) {
        if (node->getType() == MemoryInput) {
            auto cur_node = dynamic_cast<MKLDNNMemoryInputNode*>(node.get());
            if (!cur_node) {
                IE_THROW() << "Cannot cast " << node->getName() << " to MKLDNNMemoryInputNode";
            }
            auto cur_id = cur_node->getId();
            for (const auto& state : memoryStates) {
                if (state->GetName() == cur_id) {
                    auto cur_state_mem = cur_node->getStore();
                    auto data_ptr = state->GetState()->cbuffer().as<void*>();
                    auto data_size = state->GetState()->byteSize();
                    auto cur_state_mem_buf = static_cast<uint8_t*>(cur_state_mem->GetPtr());

                    cpu_memcpy(cur_state_mem_buf, data_ptr, data_size);
                }
            }
        }
    }
}

void MKLDNNPlugin::MKLDNNInferRequestBase::PullStates() {
    for (auto &node : graph->GetNodes()) {
        if (node->getType() == MemoryInput) {
            auto cur_node = dynamic_cast<MKLDNNMemoryInputNode*>(node.get());
            if (!cur_node) {
                IE_THROW() << "Cannot cast " << node->getName() << " to MKLDNNMemoryInputNode";
            }
            auto cur_id = cur_node->getId();
            for (const auto& state : memoryStates) {
                if (state->GetName() == cur_id) {
                    auto cur_state_mem = cur_node->getStore();
                    auto data_ptr = state->GetState()->cbuffer().as<void*>();
                    auto data_size = state->GetState()->byteSize();
                    auto cur_state_mem_buf = static_cast<uint8_t*>(cur_state_mem->GetPtr());

                    cpu_memcpy(data_ptr, cur_state_mem_buf, data_size);
                }
            }
        }
    }
}

void MKLDNNPlugin::MKLDNNInferRequestBase::redefineMemoryForInputNodes() {
    const auto cpuInputNodes = graph->GetInputNodesMap();

    for (const auto &blob : _inputs) {
        const auto inputNode = cpuInputNodes.find(blob.first);
        if (inputNode == cpuInputNodes.end())
            IE_THROW() << "CPU execution graph doesn't contain input node with name: " << blob.first;
        if (inputNode->second->isDynamicNode()) {
            inputNode->second->redefineOutputMemory({blob.second->getTensorDesc().getDims()});
        }
    }
}

void MKLDNNPlugin::MKLDNNInferRequestBase::InferImpl() {
    using namespace openvino::itt;
    OV_ITT_SCOPED_TASK(itt::domains::MKLDNNPlugin, profilingTask);
    auto graphLock = execNetwork->GetGraph();
    graph = &(graphLock._graph);

    ThrowIfCanceled();

    if (graph->hasDynamicInput())
        redefineMemoryForInputNodes();

    execDataPreprocessing(_inputs);

    changeDefaultPtr();

    ThrowIfCanceled();

    PushInputData();

    if (memoryStates.size() != 0) {
        PushStates();
    }

    graph->Infer(this, m_curBatch);

    if (memoryStates.size() != 0) {
        PullStates();
    }

    ThrowIfCanceled();

    graph->PullOutputData(_outputs);
}

std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> MKLDNNPlugin::MKLDNNInferRequestBase::GetPerformanceCounts() const {
    if (!graph || !graph->IsReady())
        IE_THROW() << "Graph is not ready!";
    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> perfMap;
    graph->GetPerfData(perfMap);
    return perfMap;
}

static inline void changeEdgePtr(const MKLDNNPlugin::MKLDNNEdgePtr &edge, void *newPtr) {
    edge->getMemory().GetPrimitivePtr()->set_data_handle(newPtr);
}

void MKLDNNPlugin::MKLDNNInferRequestBase::changeDefaultPtr() {
    for (auto& it : externalPtr) {
        const auto& inputNodesMap = graph->GetInputNodesMap();
        auto input = inputNodesMap.find(it.first);
        if (input != inputNodesMap.end()) {
            MKLDNNNodePtr inputNodePtr = input->second;
            if (inputNodePtr->getChildEdgeAt(0)->getMemory().GetPrimitive().get_data_handle() == it.second)
                continue;
            auto& childEdges = inputNodePtr->getChildEdges();
            // Input cannot be in-place with other primitives
            bool canBeInPlace = true;
            for (auto& childEdge : childEdges) {
                auto ce = childEdge.lock();
                if (!ce)
                    IE_THROW() << "Node " << inputNodePtr->getName() << " contains empty child edge";

                auto& child = ce->getChild();

                if (child->isConstant()) {
                    canBeInPlace = false;
                    break;
                }

                if (child->getType() == Concatenation) {
                    auto concat = dynamic_cast<MKLDNNConcatNode*>(child.get());
                    if (concat && concat->isOptimized()) {
                        canBeInPlace = false;
                        break;
                    }
                }

                // Cannot be in-place before split because split is using different ptrs without offsets
                if (child->getType() == Split) {
                    canBeInPlace = false;
                    break;
                }

                if (child->isInPlace()) {
                    canBeInPlace = false;
                    break;
                }

                auto& edges = child->getChildEdges();
                for (auto& edge : edges) {
                    auto e = edge.lock();
                    if (!e)
                        IE_THROW() << "Node " << child->getName() << " contains empty child edge";

                    if (e->getMemory().GetPrimitive().get_data_handle() == ce->getMemory().GetPrimitive().get_data_handle()) {
                        canBeInPlace = false;
                        break;
                    }
                }

                if (!canBeInPlace)
                    break;
            }
            if (canBeInPlace) {
                for (auto& edge : childEdges) {
                    auto e = edge.lock();
                    if (!e)
                        IE_THROW() << "Node " << inputNodePtr->getName() << " contains empty child edge";

                    changeEdgePtr(e, it.second);
                }
            }

            continue;
        }

        const auto& outputNodesMap = graph->GetOutputNodesMap();
        auto output = outputNodesMap.find(it.first);
        if (output != outputNodesMap.end()) {
            auto parentEdge = output->second->getParentEdgeAt(0);
            if (parentEdge->getMemory().GetPrimitive().get_data_handle() == it.second)
                continue;

            bool canBeInPlace = true;
            void* defaultPtr = parentEdge->getMemory().GetPrimitivePtr()->get_data_handle();
            // Cannot be in-place after concat because concat is using different ptrs without offsets
            auto parent = parentEdge->getParent();
            MKLDNNNodePtr previousParent;
            do {
                previousParent = parent;
                if (parent->getChildEdges().size() != 1 || parent->isConstant() || parent->isInPlace()) {
                    canBeInPlace = false;
                    break;
                }

                auto& parentEdges = parent->getParentEdges();
                for (auto& edge : parentEdges) {
                    auto e = edge.lock();
                    if (!e)
                        IE_THROW() << "Node " << parent->getName() << " contains empty parent edge";

                    if (e->getMemory().GetPrimitivePtr()->get_data_handle() == defaultPtr) {
                        parent = e->getParent();
                        break;
                    }
                }
            } while (previousParent != parent);
            if (canBeInPlace)
                changeEdgePtr(parentEdge, it.second);
            continue;
        }
        IE_THROW() << "Cannot find input/output blob: " << it.first;
    }
}


void MKLDNNPlugin::MKLDNNInferRequestBase::SetBatch(int new_batch) {
    if (!graph->getProperty().enableDynamicBatch)
        IE_THROW() << "Dynamic batch is not enabled.";

    if (new_batch < 1 || new_batch > graph->getProperty().batchLimit) {
        IE_THROW() << "Invalid dynamic batch size " << new_batch <<
            " for this request.";
    }

    m_curBatch = new_batch;

    for (const auto& node : graph->GetNodes()) {
        node->setDynamicBatchLim(new_batch);
    }
}

std::vector<InferenceEngine::IVariableStateInternal::Ptr> MKLDNNPlugin::MKLDNNInferRequestBase::QueryState() {
    return memoryStates;
}

void MKLDNNPlugin::MKLDNNInferRequestBase::SetAsyncRequest(MKLDNNAsyncInferRequest* asyncRequest) {
    _asyncRequest = asyncRequest;
}

void MKLDNNPlugin::MKLDNNInferRequestBase::ThrowIfCanceled() const {
    if (_asyncRequest != nullptr) {
        _asyncRequest->ThrowIfCanceled();
    }
}

InferenceEngine::Precision
MKLDNNPlugin::MKLDNNInferRequestBase::normToInputSupportedPrec(const std::pair<const std::string, InferenceEngine::Blob::Ptr>& input) const {
    const auto& inputTensorDesc = input.second->getTensorDesc();
    auto inPrec = inputTensorDesc.getPrecision();
    if (graph->hasMeanImageFor(input.first) && one_of(inPrec, InferenceEngine::Precision::U8, InferenceEngine::Precision::BOOL)) {
        inPrec = InferenceEngine::Precision::FP32;
    } else {
        inPrec = normalizeToSupportedPrecision(inPrec);
    }

    if (inPrec == InferenceEngine::Precision::UNSPECIFIED) {
        IE_THROW() << "Unsupported input precision " << inputTensorDesc.getPrecision();
    }

    return inPrec;
}

/* ========================================== MKLDNNLegacyInferRequest ========================================== */
MKLDNNPlugin::MKLDNNLegacyInferRequest::MKLDNNLegacyInferRequest(InferenceEngine::InputsDataMap networkInputs,
                                                                 InferenceEngine::OutputsDataMap networkOutputs,
                                                                 std::shared_ptr<MKLDNNExecNetwork> execNetwork)
: MKLDNNInferRequestBase(networkInputs, networkOutputs, execNetwork) {
    CreateInferRequest();
}

void MKLDNNPlugin::MKLDNNLegacyInferRequest::initBlobs() {
    for (const auto& it : _networkInputs) {
        MKLDNNLegacyInferRequest::GetBlob(it.first);
    }
    for (const auto& it : _networkOutputs) {
        MKLDNNLegacyInferRequest::GetBlob(it.first);
    }
}

// TODO [DS] : restore API 1.0 version, after dynamism supporting will be removed from API 1.0
void MKLDNNPlugin::MKLDNNLegacyInferRequest::SetBlob(const std::string& name, const InferenceEngine::Blob::Ptr &data) {
    OV_ITT_SCOPED_TASK(itt::domains::MKLDNNPlugin, "SetBlobOldApi");
    if (name.empty()) {
        IE_THROW(NotFound) << "Failed to set blob with empty name";
    }

    if (!data)
        IE_THROW(NotAllocated) << "Failed to set empty blob with name: \'" << name << "\'";
    InferenceEngine::InputInfo::Ptr foundInput;
    InferenceEngine::DataPtr foundOutput;
    const bool isInput = findInputAndOutputBlobByName(name, foundInput, foundOutput);

    MKLDNNNodePtr inputNode, outputNode;
    if (isInput) {
        inputNode = graph->getInputNodeByName(name);
    } else if (foundOutput) {
        outputNode = graph->getOutputNodeByName(name);
    } else {
        IE_THROW() << "Failed to set blob, can't find blob with name: " << name;
    }

    const bool compoundBlobPassed = data->is<InferenceEngine::CompoundBlob>();
    if (!compoundBlobPassed && data->buffer() == nullptr)
        IE_THROW(NotAllocated) << "Input data was not allocated. Input name: \'" << name << "\'";
    if (data->size() == 0 &&
        !((inputNode && inputNode->isDynamicNode()) || (outputNode && outputNode->isDynamicNode()))) {
        IE_THROW() << "Input data is empty. Input name: \'" << name << "\'";
    }

    size_t dataSize = data->size();
    const auto &blobDesc = data->getTensorDesc();

    if (isInput) {
        if (foundInput->getPrecision() != blobDesc.getPrecision()) {
            IE_THROW(ParameterMismatch) << "Failed to set input blob with precision: "
                               << blobDesc.getPrecision() << ", if CNNNetwork input blob precision is: " << foundInput->getPrecision();
        }

        const bool preProcRequired = preProcessingRequired(foundInput, data);
        if (compoundBlobPassed && !preProcRequired) {
            IE_THROW(NotImplemented)
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

            const bool isDynamic = inputNode->isDynamicNode();
            if (!isDynamic && dataSize != inputSize) {
                IE_THROW() << "Input blob size is not equal network input size ("
                                   << dataSize << "!=" << inputSize << ").";
            }

            if (!isDynamic && foundInput->getTensorDesc().getDims() != blobDesc.getDims()) {
                IE_THROW(ParameterMismatch) << "Failed to set input blob. Dimensions mismatch.";
            }

            if (blobDesc.getLayout() != InferenceEngine::Layout::ANY && foundInput->getTensorDesc().getLayout() != InferenceEngine::Layout::ANY) {
                if (isDynamic && InferenceEngine::TensorDesc(foundInput->getPrecision(), blobDesc.getDims(), foundInput->getLayout()).getBlockingDesc() !=
                        blobDesc.getBlockingDesc())
                    IE_THROW(ParameterMismatch) << "Failed to set input blob. Layouts mismatch.";

                if (!isDynamic && foundInput->getTensorDesc().getBlockingDesc() != blobDesc.getBlockingDesc())
                    IE_THROW(ParameterMismatch) << "Failed to set input blob. Blocking descriptor mismatch.";
            }

            MemoryDescPtr actualDesc = graph->getInputNodeByName(name)->getBaseMemDescAtOutputPort(0);
            bool blobHasAnyLayout = blobDesc.getLayout() == InferenceEngine::Layout::ANY;
            if (!blobHasAnyLayout && !actualDesc->isDefined()) {
                // we must define desc for dynamic case
                // otherwise we got incorrect check on shape compatibility inside isCompatible
                // because lower and upper bound will be compared
                actualDesc = actualDesc->cloneWithNewDims(blobDesc.getLayout() == InferenceEngine::Layout::SCALAR ? InferenceEngine::SizeVector{1} :
                                                                                                                    blobDesc.getDims());
            }
            if (!blobHasAnyLayout &&
                actualDesc->isCompatible(MemoryDescUtils::convertToCpuBlockedMemoryDesc(blobDesc)) &&
                    graph->_normalizePreprocMap.find(name) == graph->_normalizePreprocMap.end() && !graph->getProperty().batchLimit) {
                externalPtr[name] = data->buffer();
            } else if (externalPtr.find(name) != externalPtr.end()) {
                externalPtr.erase(name);
            }
            _inputs[name] = data;
        }
    }
    if (foundOutput) {
        if (compoundBlobPassed) {
            IE_THROW(NotImplemented)
                               << "cannot set compound blob: supported only for input pre-processing";
        }
        if (foundOutput->getPrecision() != blobDesc.getPrecision()) {
            IE_THROW(ParameterMismatch) << "Failed to set output blob with precision: "
                               << blobDesc.getPrecision() << ", if CNNNetwork output blob precision is: " << foundOutput->getPrecision();
        }

        const bool isDynamic = outputNode->isDynamicNode();
        if (!isDynamic) {
            size_t outputSize = foundOutput->getTensorDesc().getLayout() != InferenceEngine::Layout::SCALAR
                ? InferenceEngine::details::product(foundOutput->getDims())
                : 1;
            if (dataSize != outputSize) {
                IE_THROW() << "Output blob size is not equal network output size ("
                                   << dataSize << "!=" << outputSize << ").";
            }
        }
        if (!isDynamic && foundOutput->getTensorDesc().getDims() != blobDesc.getDims()) {
            IE_THROW(ParameterMismatch) << "Failed to set output Blob. Dimensions mismatch.";
        }

        if (blobDesc.getLayout() != InferenceEngine::Layout::ANY && foundOutput->getTensorDesc().getLayout() != InferenceEngine::Layout::ANY) {
            if (isDynamic && InferenceEngine::TensorDesc(foundOutput->getPrecision(), blobDesc.getDims(), foundOutput->getLayout()).getBlockingDesc() !=
                    blobDesc.getBlockingDesc())
                IE_THROW(ParameterMismatch) << "Failed to set input blob. Layouts mismatch.";

            if (!isDynamic && foundOutput->getTensorDesc().getBlockingDesc() != blobDesc.getBlockingDesc())
                IE_THROW(ParameterMismatch) << "Failed to set output blob. Blocking descriptor mismatch.";
        }

        const auto &desc = graph->getOutputNodeByName(name)->getParentEdgesAtPort(0)[0]->getMemory().getDesc();
        if (!isDynamic && blobDesc == MemoryDescUtils::convertToTensorDesc(desc) && !graph->getProperty().batchLimit) {
            externalPtr[name] = data->buffer();
        } else if (externalPtr.find(name) != externalPtr.end()) {
            externalPtr.erase(name);
        }
        _outputs[name] = data;
    }
}

// TODO [DS] : restore API 1.0 version, after dynamism supporting will be removed from API 1.0
InferenceEngine::Blob::Ptr MKLDNNPlugin::MKLDNNLegacyInferRequest::GetBlob(const std::string& name) {
    OV_ITT_SCOPED_TASK(itt::domains::MKLDNNPlugin, "GetBlobOldApi");

    if (!graph || !graph->IsReady())
        IE_THROW() << "Graph is not ready!";

    InferenceEngine::Blob::Ptr data;

    const auto &inMap = graph->inputNodesMap;
    auto input = inMap.find(name);
    if (input != inMap.end()) {
        // ROI blob is returned only if it was set previously.
        auto it = _preProcData.find(name);
        if (it != _preProcData.end()) {
            data = it->second->getRoiBlob();
            return data;
        }

        if (_inputs.find(name) == _inputs.end()) {
            if (_networkInputs.find(name) != _networkInputs.end()) {
                InferenceEngine::TensorDesc desc = _networkInputs[name]->getTensorDesc();
                bool isDynamic = input->second->isDynamicNode();

                _inputs[name] = make_blob_with_precision(desc);
                _inputs[name]->allocate();

                if (!isDynamic &&
                    desc == MemoryDescUtils::convertToTensorDesc(graph->getInputNodeByName(name)->getChildEdgesAtPort(0)[0]->getMemory().getDesc()) &&
                        graph->_normalizePreprocMap.find(name) == graph->_normalizePreprocMap.end() && !graph->getProperty().batchLimit) {
                    externalPtr[name] = _inputs[name]->buffer();
                }
            } else {
                IE_THROW() << "Blob with name: " << name << " exists in MKLDNN graph, but absents in network inputs";
            }
        }
        data = _inputs[name];
        checkBlob(data, name, true);

        // check if preprocess required, but still wasn't set
        auto preProcessedInput = std::find_if(std::begin(_networkInputs), std::end(_networkInputs),
            [&](const std::pair<std::string, InferenceEngine::InputInfo::Ptr>& pair) {
                return pair.first == name;
            });
        if (preProcessedInput != std::end(_networkInputs)) {
            InferenceEngine::InputInfo::Ptr foundInput;
            InferenceEngine::DataPtr foundOutput;
            if (!findInputAndOutputBlobByName(name, foundInput, foundOutput)) {
                IE_THROW() << "Blob with name: " << name << " absents in network inputs";
            }
            if (preProcessingRequired(foundInput, data)) {
                _preProcData.emplace(name, InferenceEngine::CreatePreprocDataHelper());
                _preProcData[name]->isApplicable(data, _inputs[name]);
                _preProcData[name]->setRoiBlob(data);
            }
        }
    }

    if (graph->hasOutputWithName(name)) {
        if (auto outNode = graph->getOutputNodeByName(name)) {
            if (_outputs.find(name) == _outputs.end()) {
                if (_networkOutputs.find(name) != _networkOutputs.end()) {
                    bool isDynamic = outNode->isDynamicNode();
                    const auto &desc = outNode->getParentEdgesAtPort(0)[0]->getMemory().getDesc();

                    if (!data) {
                        InferenceEngine::TensorDesc desc = _networkOutputs[name]->getTensorDesc();
                        desc.setPrecision(normalizeToSupportedPrecision(desc.getPrecision()));

                        data = make_blob_with_precision(desc);
                        data->allocate();
                    } else {
                        const auto &expectedTensorDesc = isDynamic ? InferenceEngine::TensorDesc(desc.getPrecision(),
                                                                                                 InferenceEngine::TensorDesc::getLayoutByRank(
                                                                                                         desc.getShape().getRank()))
                                                                   : MemoryDescUtils::convertToTensorDesc(desc);
                        const auto &tensorDesc = data->getTensorDesc();
                        if (expectedTensorDesc.getPrecision() != normalizeToSupportedPrecision(tensorDesc.getPrecision())) {
                            IE_THROW(ParameterMismatch)
                                    << "Network input and output use the same name: " << name << " but expect blobs with different precision: "
                                    << tensorDesc.getPrecision() << " for input and " << expectedTensorDesc.getPrecision()
                                    << " for output.";
                        }

                        if (expectedTensorDesc.getDims() != tensorDesc.getDims()) {
                            IE_THROW(ParameterMismatch) << "Network input and output use the same name: " << name << " but expect blobs with different shapes.";
                        }

                        if (tensorDesc.getLayout() != InferenceEngine::Layout::ANY && expectedTensorDesc.getLayout() != InferenceEngine::Layout::ANY) {
                            if (tensorDesc.getLayout() != expectedTensorDesc.getLayout() &&
                                !(tensorDesc.getLayout() == InferenceEngine::Layout::BLOCKED &&
                                    InferenceEngine::TensorDesc(tensorDesc.getPrecision(),
                                                                tensorDesc.getDims(),
                                                                tensorDesc.getBlockingDesc()).getLayout() == expectedTensorDesc.getLayout())) {
                                IE_THROW(ParameterMismatch) << "Network input and output use the same name: " << name << " but expect blobs" <<
                                                            " with different layouts.";
                            }

                            if (expectedTensorDesc.getBlockingDesc() != tensorDesc.getBlockingDesc())
                                IE_THROW(ParameterMismatch) << "Network input and output use the same name: " << name
                                                            << " but expect blobs with different blocking descriptors.";
                        }
                    }

                    _outputs[name] = data;
                    if (!isDynamic && !externalPtr.count(name) && data->getTensorDesc() == MemoryDescUtils::convertToTensorDesc(desc) &&
                        !graph->getProperty().batchLimit) {
                        externalPtr[name] = data->buffer();
                    }
                } else {
                    IE_THROW() << "Blob with name: " << name << " exists in MKLDNN graph, but absents in network outputs";
                }
            }

            data = _outputs[name];
            if (!outNode->isDynamicNode())
                checkBlob(data, name, false);
        } else {
            IE_THROW() << "Output node with name: " << name << " has not been created";
        }
    }
    if (!data) {
        IE_THROW() << "Cannot find blob with name: " << name;
    }
    return data;
}

void MKLDNNPlugin::MKLDNNLegacyInferRequest::PushInputData() {
    for (auto input : _inputs) {
        auto inputName = input.first;
        if (!_networkInputs[inputName]) {
            IE_THROW() << "Input blobs map contains not registered during IInferencePlugin::LoadNetwork blob with name " << inputName;
        }

        // User can initialize input via setBlob API using tensorDesc with default (ANY) layout.
        // Currently IE doesn't specify behavior in such scenario, so we assume real layout is equal to the network input.
        auto inputBlob = input.second;
        if (inputBlob->getTensorDesc().getLayout() == InferenceEngine::ANY) {
            inputBlob->getTensorDesc().setLayout(_networkInputs[inputName]->getLayout());
        }

        pushInput(inputName, inputBlob, normToInputSupportedPrec(input));
    }
}

/* ========================================== MKLDNNInferRequest ========================================== */
MKLDNNPlugin::MKLDNNInferRequest::MKLDNNInferRequest(const std::vector<std::shared_ptr<const ov::Node>>& inputs,
                                                                 const std::vector<std::shared_ptr<const ov::Node>>& outputs,
                                                                 MKLDNNExecNetwork::Ptr execNetwork)
: MKLDNNInferRequestBase(inputs, outputs, execNetwork) {
    for (const std::shared_ptr<const ov::Node>& in : inputs) {
        modelInputsMap[ngraph::op::util::get_ie_output_name(ngraph::Output<const ngraph::Node>(in))] = in;
    }
    for (const std::shared_ptr<const ov::Node>& out : outputs) {
        modelOutputsMap[ngraph::op::util::get_ie_output_name(out->input_value(0))] = out;
    }

    CreateInferRequest();
}

void MKLDNNPlugin::MKLDNNInferRequest::initBlobs() {
    for (const auto& it : modelInputsMap) {
        MKLDNNInferRequest::GetBlob(it.first);
    }
    for (const auto& it : modelOutputsMap) {
        MKLDNNInferRequest::GetBlob(it.first);
    }
}

void MKLDNNPlugin::MKLDNNInferRequest::SetBlob(const std::string& name, const InferenceEngine::Blob::Ptr &data) {
    OV_ITT_SCOPED_TASK(itt::domains::MKLDNNPlugin, "SetBlobNewApi");
    if (name.empty()) {
        IE_THROW(NotFound) << "Failed to set blob with empty name";
    }

    if (!data)
        IE_THROW(NotAllocated) << "Failed to set empty blob with name: \'" << name << "\'";

    bool isInput = false;
    const auto inputNodeItr = modelInputsMap.find(name);
    const auto outputNodeItr = modelOutputsMap.find(name);

    if (inputNodeItr != modelInputsMap.end()) {
        if (!inputNodeItr->second) {
            IE_THROW() << "Can't SetBlob with name: " << name << ", because has null pointer to input node";
        }
        isInput = true;
    } else if (outputNodeItr != modelOutputsMap.end()) {
        if (!outputNodeItr->second) {
            IE_THROW() << "Can't SetBlob with name: " << name << ", because has null pointer to output node";
        }
        isInput = false;
    } else {
        IE_THROW(NotFound) << "Can't SetBlob with name: " << name << ", because input/output with this name doesn't exist";
    }

    const bool compoundBlobPassed = data->is<InferenceEngine::CompoundBlob>();
    if (!compoundBlobPassed && data->buffer() == nullptr)
        IE_THROW(NotAllocated) << "Input data was not allocated. Input name: \'" << name << "\'";

    const auto &blobDesc = data->getTensorDesc();

    if (isInput) {
        const auto netInPrc = InferenceEngine::details::convertPrecision(inputNodeItr->second->get_output_element_type(0));
        if (netInPrc != blobDesc.getPrecision()) {
            IE_THROW(ParameterMismatch) << "Failed to set input blob with precision: "
                               << blobDesc.getPrecision() << ", if CNNNetwork input blob precision is: " << netInPrc;
        }

        const auto shape = inputNodeItr->second->get_output_partial_shape(0);
        const bool isDynamic = shape.is_dynamic();
        if (!shape.compatible(ov::PartialShape(data->getTensorDesc().getDims()))) {
            IE_THROW() << "Can't SetBlob with name: " << name << ", because model input and blob are incompatible";
        }

        if (!isDynamic && ngraph::shape_size(shape.to_shape()) != data->size()) {
            IE_THROW() << "Can't SetBlob with name: " << name << ", because model input and blob have different size";
        }

        MemoryDescPtr actualDesc = graph->getInputNodeByName(name)->getBaseMemDescAtOutputPort(0);
        if (!actualDesc->isDefined()) {
            // we must define desc for dynamic case
            // otherwise we got incorrect check on shape compatibility inside isCompatible
            // because lower and upper bound will be compared
            actualDesc = actualDesc->cloneWithNewDims(blobDesc.getLayout() == InferenceEngine::Layout::SCALAR ? InferenceEngine::SizeVector{1} :
                                                                                                                blobDesc.getDims());
        }
        if (actualDesc->isCompatible(MemoryDescUtils::convertToCpuBlockedMemoryDesc(blobDesc)) &&
                graph->_normalizePreprocMap.find(name) == graph->_normalizePreprocMap.end() && !graph->getProperty().batchLimit) {
            externalPtr[name] = data->buffer();
        } else if (externalPtr.find(name) != externalPtr.end()) {
            externalPtr.erase(name);
        }
        _inputs[name] = data;
    } else {
        if (compoundBlobPassed) {
            IE_THROW(NotImplemented) << "cannot set compound blob: supported only for input pre-processing";
        }
        const auto netOutPrc = InferenceEngine::details::convertPrecision(outputNodeItr->second->get_input_element_type(0));
        if (netOutPrc != blobDesc.getPrecision()) {
            IE_THROW(ParameterMismatch) << "Failed to set input blob with precision: "
                               << blobDesc.getPrecision() << ", if CNNNetwork output blob precision is: " << netOutPrc;
        }

        const auto shape = outputNodeItr->second->get_input_partial_shape(0);
        const bool isDynamic = shape.is_dynamic();

        if (!shape.compatible(ov::PartialShape(data->getTensorDesc().getDims()))) {
            IE_THROW() << "Can't SetBlob with name: " << name << ", because model output and blob are incompatible";
        }

        if (!isDynamic && ngraph::shape_size(shape.to_shape()) != data->size()) {
            IE_THROW() << "Can't SetBlob with name: " << name << ", because model output and blob have different size";
        }

        const auto &desc = graph->getOutputNodeByName(name)->getParentEdgesAtPort(0)[0]->getMemory().getDesc();
        if (!isDynamic && blobDesc == MemoryDescUtils::convertToTensorDesc(desc) && !graph->getProperty().batchLimit) {
            externalPtr[name] = data->buffer();
        } else if (externalPtr.find(name) != externalPtr.end()) {
            externalPtr.erase(name);
        }
        _outputs[name] = data;
    }
}

InferenceEngine::Blob::Ptr MKLDNNPlugin::MKLDNNInferRequest::GetBlob(const std::string& name) {
    OV_ITT_SCOPED_TASK(itt::domains::MKLDNNPlugin, "GetBlobNewApi");

    if (!graph || !graph->IsReady())
        IE_THROW() << "Graph is not ready!";

    InferenceEngine::Blob::Ptr data;

    const auto &inMap = graph->inputNodesMap;
    auto input = inMap.find(name);
    if (input != inMap.end()) {
        if (_inputs.find(name) == _inputs.end()) {
            auto inputNode = modelInputsMap.find(name);
            if (inputNode != modelInputsMap.end()) {
                if (!inputNode->second) {
                    IE_THROW() << "Can't GetBlob with name: " << name << ", because has null pointer to input node";
                }

                const auto shape = inputNode->second->get_output_partial_shape(0);
                const bool isDynamic = shape.is_dynamic();
                InferenceEngine::SizeVector dims;
                if (isDynamic) {
                    dims = InferenceEngine::SizeVector(shape.rank().get_length(), 0);
                } else {
                    dims = shape.to_shape();
                }

                InferenceEngine::TensorDesc desc(InferenceEngine::details::convertPrecision(inputNode->second->get_output_element_type(0)),
                                                 dims, InferenceEngine::TensorDesc::getLayoutByRank(dims.size()));

                _inputs[name] = make_blob_with_precision(desc);
                _inputs[name]->allocate();

                if (!isDynamic &&
                    desc == MemoryDescUtils::convertToTensorDesc(graph->getInputNodeByName(name)->getChildEdgesAtPort(0)[0]->getMemory().getDesc()) &&
                        graph->_normalizePreprocMap.find(name) == graph->_normalizePreprocMap.end() && !graph->getProperty().batchLimit) {
                    externalPtr[name] = _inputs[name]->buffer();
                }
            } else {
                IE_THROW() << "Blob with name: " << name << " exists in MKLDNN graph, but absents in network inputs";
            }
        }
        data = _inputs[name];
    }

    const auto &outMap = graph->outputNodesMap;
    auto output = outMap.find(name);
    if (output != outMap.end()) {
        if (_outputs.find(name) == _outputs.end()) {
            auto outputNode = modelOutputsMap.find(name);
            if (modelOutputsMap.find(name) != modelOutputsMap.end()) {
                const auto shape = outputNode->second->get_input_partial_shape(0);
                bool isDynamic = shape.is_dynamic();

                if (!data) {
                    InferenceEngine::SizeVector dims;
                    if (isDynamic) {
                        dims = InferenceEngine::SizeVector(shape.rank().get_length(), 0);
                    } else {
                        dims = shape.to_shape();
                    }

                    InferenceEngine::TensorDesc desc(InferenceEngine::details::convertPrecision(outputNode->second->get_input_element_type(0)),
                                                     dims, InferenceEngine::TensorDesc::getLayoutByRank(dims.size()));

                    data = make_blob_with_precision(desc);
                    data->allocate();
                } else {
                    if (!shape.compatible(ov::PartialShape(data->getTensorDesc().getDims()))) {
                        IE_THROW(ParameterMismatch) << "Network input and output use the same name: " << name << ", but expect blobs with different shapes.";
                    }

                    const auto netOutPrc = InferenceEngine::details::convertPrecision(outputNode->second->get_input_element_type(0));
                    if (netOutPrc != data->getTensorDesc().getPrecision()) {
                        IE_THROW(ParameterMismatch)
                                    << "Network input and output use the same name: " << name << " but expect blobs with different precision: "
                                    << data->getTensorDesc().getPrecision() << " for input and " << netOutPrc
                                    << " for output.";
                    }
                }

                _outputs[name] = data;
                if (!isDynamic && !externalPtr.count(name) &&
                    data->getTensorDesc() == MemoryDescUtils::convertToTensorDesc(output->second->getParentEdgesAtPort(0)[0]->getMemory().getDesc()) &&
                        !graph->getProperty().batchLimit) {
                    externalPtr[name] = data->buffer();
                }
            } else {
                IE_THROW() << "Blob with name: " << name << " exists in MKLDNN graph, but absents in network outputs";
            }
        }
        data = _outputs[name];
    }

    if (!data) {
        IE_THROW() << "Cannot find blob with name: " << name;
    }

    return data;
}

void MKLDNNPlugin::MKLDNNInferRequest::PushInputData() {
    for (auto input : _inputs) {
        auto inputName = input.first;
        if (!modelInputsMap[inputName]) {
            IE_THROW() << "Input blobs map contains not registered during IInferencePlugin::LoadNetwork blob with name " << inputName;
        }

        pushInput(inputName, input.second, normToInputSupportedPrec(input));
    }
}
