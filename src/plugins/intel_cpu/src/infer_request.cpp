// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "infer_request.h"
#include "dnnl_extension_utils.h"
#include <vector>
#include <string>
#include <map>
#include <blob_factory.hpp>
#include "nodes/concat.h"
#include "nodes/split.h"
#include <ie_compound_blob.h>
#include <ie_common.h>
#include "exec_network.h"
#include "itt.h"
#include "nodes/common/cpu_convert.h"
#include "memory_state.h"
#include "nodes/memory.hpp"
#include "nodes/common/cpu_memcpy.h"
#include "async_infer_request.h"
#include <debug.h>
#include "utils/general_utils.h"
#include "utils/cpu_utils.hpp"
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include <transformations/utils/utils.hpp>
#include <ie_ngraph_utils.hpp>
#include "proxy_mem_mgr.h"
#include "openvino/runtime/make_tensor.hpp"
#include <utils/general_utils.h>

namespace ov {
namespace intel_cpu {

void InferRequestBase::CreateInferRequest() {
    auto id = (execNetwork->_numRequests)++;
    profilingTask = openvino::itt::handle("INTEL_CPU_INFER_" + execNetwork->_name + "_" + std::to_string(id));

    if (execNetwork->_graphs.size() == 0)
        IE_THROW() << "No graph was found";
    graph = &(execNetwork->GetGraph()._graph);

    initBlobs();

    // Save all MemoryLayer data tensors. Will use insight about mechanics
    // of MemoryLayer implementation. It uses output edge of MemoryLayer
    // producer as storage for tensor to keep it between infer calls.
    for (auto& node : graph->GetNodes()) {
        if (node->getType() == Type::MemoryInput) {
            auto memoryNode = dynamic_cast<node::MemoryInput*>(node.get());
            if (!memoryNode) {
                IE_THROW() << "Cannot cast " << node->getName() << " to MemoryInput";
            }
            auto state_store = memoryNode->getStore();
            auto state_name = memoryNode->getId();

            // Remove suffix with pair ID. Internal information.
            auto suffix_idx = state_name.find("/id=");
            if (suffix_idx != std::string::npos)
                state_name = state_name.substr(0, suffix_idx);

            memoryStates.emplace_back(new VariableState(state_name, state_store));
        }
    }
}

InferRequestBase::~InferRequestBase() {
    --(execNetwork->_numRequests);
}

void InferRequestBase::pushInput(const std::string& inputName, InferenceEngine::Blob::Ptr& inputBlob, InferenceEngine::Precision inPrec) {
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

void InferRequestBase::PushStates() {
    for (auto &node : graph->GetNodes()) {
        if (node->getType() == Type::MemoryInput) {
            auto cur_node = dynamic_cast<node::MemoryInput*>(node.get());
            if (!cur_node) {
                IE_THROW() << "Cannot cast " << node->getName() << " to MemoryInput";
            }
            auto cur_id = cur_node->getId();
            for (const auto& state : memoryStates) {
                if (state->GetName() == cur_id) {
                    auto cur_state_mem = cur_node->getStore();
                    auto data_ptr = state->GetState()->cbuffer().as<void*>();
                    auto data_size = state->GetState()->byteSize();
                    auto cur_state_mem_buf = static_cast<uint8_t*>(cur_state_mem->getData());

                    cpu_memcpy(cur_state_mem_buf, data_ptr, data_size);
                }
            }
        }
    }
}

void InferRequestBase::PullStates() {
    for (auto &node : graph->GetNodes()) {
        if (node->getType() == Type::MemoryInput) {
            auto cur_node = dynamic_cast<node::MemoryInput*>(node.get());
            if (!cur_node) {
                IE_THROW() << "Cannot cast " << node->getName() << " to MemoryInput";
            }
            auto cur_id = cur_node->getId();
            for (const auto& state : memoryStates) {
                if (state->GetName() == cur_id) {
                    auto cur_state_mem = cur_node->getStore();
                    auto data_ptr = state->GetState()->cbuffer().as<void*>();
                    auto data_size = state->GetState()->byteSize();
                    auto cur_state_mem_buf = static_cast<uint8_t*>(cur_state_mem->getData());

                    cpu_memcpy(data_ptr, cur_state_mem_buf, data_size);
                }
            }
        }
    }
}

void InferRequestBase::redefineMemoryForInputNodes() {
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

void InferRequestBase::InferImpl() {
    using namespace openvino::itt;
    OV_ITT_SCOPED_TASK(itt::domains::intel_cpu, profilingTask);
    auto graphLock = execNetwork->GetGraph();
    graph = &(graphLock._graph);

    ThrowIfCanceled();
    convertBatchedInputBlobs();

    if (graph->hasDynamicInput()) {
        redefineMemoryForInputNodes();
    }

    execDataPreprocessing(_inputs);

    changeDefaultPtr();

    ThrowIfCanceled();

    PushInputData();

    if (memoryStates.size() != 0) {
        PushStates();
    }

    graph->Infer(this);

    if (memoryStates.size() != 0) {
        PullStates();
    }

    ThrowIfCanceled();

    // update output control blocks, if any, in order to refresh internal buffers
    if (Graph::Status::ReadyDynamic == graph->getStatus()) {
        for (auto&& item : outputControlBlocks) {
            item.second.update();
        }
    }

    graph->PullOutputData(_outputs);
}

std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> InferRequestBase::GetPerformanceCounts() const {
    if (!graph || !graph->IsReady())
        IE_THROW() << "Graph is not ready!";
    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> perfMap;
    graph->GetPerfData(perfMap);
    return perfMap;
}

static inline void changeEdgePtr(const EdgePtr &edge, InferenceEngine::Blob::Ptr blob) {
    auto size = blob->byteSize();
    auto& mem = edge->getMemory();
    auto memMngr = mem.getMemoryMngr();
    IE_ASSERT(memMngr);
    memMngr->setExtBuff(blob->buffer(), size);
}

void InferRequestBase::changeDefaultPtr() {
    const auto& inputNodesMap = graph->GetInputNodesMap();
    const auto& outputNodesMap = graph->GetOutputNodesMap();
    std::unordered_set<const void*> inputPtrs;
    std::function<void(const EdgePtr &edge, InferenceEngine::Blob::Ptr blob)> changeInpPtr;
    if (Graph::Status::ReadyDynamic == graph->getStatus()) {
        changeInpPtr = [&inputPtrs](const EdgePtr &edge, InferenceEngine::Blob::Ptr blob) {
            changeEdgePtr(edge, blob);
            inputPtrs.insert(blob->buffer());
        };
    } else {
        changeInpPtr = [](const EdgePtr &edge, InferenceEngine::Blob::Ptr blob) {
            changeEdgePtr(edge, blob);
        };
    }

    for (auto& it : externalPtr) {
        auto input = inputNodesMap.find(it.first);
        if (inputNodesMap.end() == input) {
            OPENVINO_ASSERT(outputNodesMap.count(it.first), "Cannot find input/output blob: ", it.first);
            continue;
        }
        NodePtr inputNodePtr = input->second;
        if (inputNodePtr->getChildEdgeAt(0)->getMemory().getData() == static_cast<void*>(it.second->buffer()))
            continue;
        auto& childEdges = inputNodePtr->getChildEdges();
        // Perform checks that the user's memory will not be modified
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

            // the input memory should be referenced by the children, otherwise it should be written to a
            // specific location
            if (ce->inPlace(Edge::LOOK_DOWN)) {
                canBeInPlace = false;
                break;
            }

            if (auto result = ce->modifiedInPlace()) {
                canBeInPlace = false;
                break;
            }

            if (child->getType() == Type::Concatenation && child->isInPlace()) {
                canBeInPlace = false;
                break;
            }
        }
        if (canBeInPlace) {
            for (auto& edge : childEdges) {
                auto e = edge.lock();
                if (!e)
                    IE_THROW() << "Node " << inputNodePtr->getName() << " contains empty child edge";

                changeInpPtr(e, it.second);
            }
        }
    }

    for (auto& it : externalPtr) {
        const auto& name = it.first;
        auto output = outputNodesMap.find(name);
        if (outputNodesMap.end() == output) {
            continue;
        }
        auto parentEdge = output->second->getParentEdgeAt(0);

        if (parentEdge->getMemory().getData() == static_cast<void*>(it.second->buffer()))
            continue;

        bool canBeInPlace = true;
        void* defaultPtr = parentEdge->getMemory().getData();
        // Cannot be in-place after concat because concat is using different ptrs without offsets
        auto parent = parentEdge->getParent();
        NodePtr previousParent;
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

                if (e->getMemory().getData() == defaultPtr) {
                    parent = e->getParent();
                    break;
                }
            }
        } while (previousParent != parent);
        if (canBeInPlace)
            changeEdgePtr(parentEdge, it.second);
    }

    if (Graph::Status::ReadyDynamic == graph->getStatus()) {
        const auto &outMemMngrMap = graph->outputNodesMemMngrMap;
        for (auto&& item : outMemMngrMap) {
            const auto& name = item.first;

            // share intel_cpu::Tensor to Graph by injecting to corresponding ProxyMemoryMngr instance.
            auto outputMemMngr = item.second;
            OPENVINO_ASSERT(outputMemMngr, "proxy mem manager for output ", name, " is empty.");

            auto controlBlockItr = outputControlBlocks.find(name);

            if (controlBlockItr != outputControlBlocks.end()) {
                auto output = outputNodesMap.find(name);
                OPENVINO_ASSERT(outputNodesMap.end() != output, "Node with name: ", name, " is absent in the outputNodesMap");
                auto parentEdge = output->second->getParentEdgeAt(0);
                //avoid cyclic memory use
                auto&& controlBlock = controlBlockItr->second;

                std::shared_ptr<IMemoryMngr> memMngr = inputPtrs.count(controlBlock.rawPtr()) ? // same memory is used on the input and output
                    controlBlock.nextMemMngr() : // then swap internal buffer to avoid data corruption
                    controlBlock.currentMemMngr(); // else reuse the existing buffer

                outputMemMngr->setMemMngr(memMngr);
                DEBUG_LOG("reset proxy ", outputMemMngr, ", actual ", controlBlock.currentMemMngr(), " graph ", graph, " inferrequest ", this);
                DEBUG_LOG(name, ", blob ", controlBlock.blob(), ", tensor ", controlBlock.tensor());
            } else {
                outputMemMngr->reset(); // switch to the internal memory since memory sharing is no longer possible
            }
        }
    }
}

std::vector<InferenceEngine::IVariableStateInternal::Ptr> InferRequestBase::QueryState() {
    return memoryStates;
}

void InferRequestBase::SetAsyncRequest(AsyncInferRequest* asyncRequest) {
    _asyncRequest = asyncRequest;
}

void InferRequestBase::ThrowIfCanceled() const {
    if (_asyncRequest != nullptr) {
        _asyncRequest->ThrowIfCanceled();
    }
}

InferenceEngine::Precision
InferRequestBase::normToInputSupportedPrec(const std::pair<const std::string, InferenceEngine::Blob::Ptr>& input) const {
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

/* ========================================== LegacyInferRequest ========================================== */
LegacyInferRequest::LegacyInferRequest(InferenceEngine::InputsDataMap networkInputs,
                                       InferenceEngine::OutputsDataMap networkOutputs,
                                       std::shared_ptr<ExecNetwork> execNetwork)
    : InferRequestBase(networkInputs, networkOutputs, execNetwork) {
    CreateInferRequest();
}

void LegacyInferRequest::initBlobs() {
    for (const auto& it : _networkInputs) {
        LegacyInferRequest::GetBlob(it.first);
    }
    for (const auto& it : _networkOutputs) {
        LegacyInferRequest::GetBlob(it.first);
    }
}

void LegacyInferRequest::changeDefaultPtr() {
    // renew external pointers before infer
    const auto &inMap = graph->inputNodesMap;
    for (auto &it : inMap) {
        const auto &name = it.first;
        auto itr = externalPtr.find(name);
        if (itr != externalPtr.end() && !(itr->second->buffer() == _inputs[name]->buffer())) {
            itr->second = _inputs[name];
        }
    }
    const auto &outMap = graph->outputNodesMap;
    for (auto &it : outMap) {
        const auto &name = it.first;
        auto itr = externalPtr.find(name);
        if (itr != externalPtr.end() && !(itr->second->buffer() == _outputs[name]->buffer())) {
            itr->second = _outputs[name];
        }
    }
    InferRequestBase::changeDefaultPtr();
}

void LegacyInferRequest::SetBlob(const std::string& name, const InferenceEngine::Blob::Ptr &data) {
    OV_ITT_SCOPED_TASK(itt::domains::intel_cpu, "SetBlobLegacy");
    if (name.empty()) {
        IE_THROW(NotFound) << "Failed to set blob with empty name";
    }

    if (!data)
        IE_THROW(NotAllocated) << "Failed to set empty blob with name: \'" << name << "\'";
    const bool compoundBlobPassed = data->is<InferenceEngine::CompoundBlob>();
    if (!compoundBlobPassed && data->buffer() == nullptr)
        IE_THROW(NotAllocated) << "Input data was not allocated. Input name: \'" << name << "\'";
    if (data->size() == 0) {
        IE_THROW() << "Input data is empty. Input name: \'" << name << "\'";
    }

    InferenceEngine::InputInfo::Ptr foundInput;
    InferenceEngine::DataPtr foundOutput;
    size_t dataSize = data->size();
    findInputAndOutputBlobByName(name, foundInput, foundOutput);

    if (foundInput) {
        if (foundInput->getPrecision() != data->getTensorDesc().getPrecision()) {
            IE_THROW(ParameterMismatch) << "Failed to set input blob with precision: "
                               << data->getTensorDesc().getPrecision() << ", if CNNNetwork input blob precision is: " << foundInput->getPrecision();
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
            if (dataSize != inputSize) {
                IE_THROW() << "Input blob size is not equal network input size ("
                                   << dataSize << "!=" << inputSize << ").";
            }

            if (foundInput->getTensorDesc().getDims() != data->getTensorDesc().getDims()) {
                IE_THROW(ParameterMismatch) << "Failed to set input blob. Dimensions mismatch.";
            }

            if (data->getTensorDesc().getLayout() != InferenceEngine::Layout::ANY && foundInput->getTensorDesc().getLayout() != InferenceEngine::Layout::ANY &&
                foundInput->getTensorDesc().getBlockingDesc() != data->getTensorDesc().getBlockingDesc()) {
                IE_THROW(ParameterMismatch) << "Failed to set input blob. Blocking descriptor mismatch.";
            }

            auto pBlobDesc = MemoryDescUtils::interpretAsBlobDesc(graph->getInputNodeByName(name)->getChildEdgesAtPort(0)[0]->getMemory());
            if (data->getTensorDesc() == pBlobDesc &&
                graph->_normalizePreprocMap.find(name) == graph->_normalizePreprocMap.end()) {
                externalPtr[name] = data;
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
        if (foundOutput->getPrecision() != data->getTensorDesc().getPrecision()) {
            IE_THROW(ParameterMismatch) << "Failed to set output blob with precision: "
                               << data->getTensorDesc().getPrecision() << ", if CNNNetwork output blob precision is: " << foundOutput->getPrecision();
        }
        size_t outputSize = foundOutput->getTensorDesc().getLayout() != InferenceEngine::Layout::SCALAR
            ? InferenceEngine::details::product(foundOutput->getDims())
            : 1;
        if (dataSize != outputSize) {
            IE_THROW() << "Output blob size is not equal network output size ("
                               << dataSize << "!=" << outputSize << ").";
        }
        if (foundOutput->getTensorDesc().getDims() != data->getTensorDesc().getDims()) {
            IE_THROW(ParameterMismatch) << "Failed to set output Blob. Dimensions mismatch.";
        }
        if (data->getTensorDesc().getLayout() != InferenceEngine::Layout::ANY && foundOutput->getTensorDesc().getLayout() != InferenceEngine::Layout::ANY &&
            foundOutput->getTensorDesc().getBlockingDesc() != data->getTensorDesc().getBlockingDesc()) {
                IE_THROW(ParameterMismatch) << "Failed to set output blob. Blocking descriptor mismatch.";
        }

        auto pBlobDesc = MemoryDescUtils::interpretAsBlobDesc(graph->getOutputNodeByName(name)->getParentEdgesAtPort(0)[0]->getMemory());
        if (data->getTensorDesc() == pBlobDesc) {
            externalPtr[name] = data;
        } else if (externalPtr.find(name) != externalPtr.end()) {
            externalPtr.erase(name);
        }
        _outputs[name] = data;
    }
}

InferenceEngine::Blob::Ptr LegacyInferRequest::GetBlob(const std::string& name) {
    OV_ITT_SCOPED_TASK(itt::domains::intel_cpu, "GetBlobLegacy");

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
            auto pBlob = MemoryDescUtils::interpretAsBlob(graph->getInputNodeByName(name)->getChildEdgesAtPort(0)[0]->getMemory());
            if (!pBlob) {
                IE_THROW() << "Can not interpret cpu plugin memory object as InferenceEngine::Blob. Input node name: " << name;
            }

            InferenceEngine::TensorDesc desc = pBlob->getTensorDesc();
            auto itr = _networkInputs.find(name);
            if (itr != _networkInputs.end()) {
                const InferenceEngine::Layout &l = itr->second->getLayout();
                const InferenceEngine::Precision &p = itr->second->getPrecision();
                const InferenceEngine::SizeVector &dims = itr->second->getTensorDesc().getDims();
                desc = InferenceEngine::TensorDesc(p, dims, l);
            }

            _inputs[name] = make_blob_with_precision(desc);
            _inputs[name]->allocate();
            if (pBlob->getTensorDesc() == desc &&
                graph->_normalizePreprocMap.find(name) == graph->_normalizePreprocMap.end()) {
                externalPtr[name] = _inputs[name];
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
        if (_outputs.find(name) == _outputs.end()) {
            auto pBlobDesc = MemoryDescUtils::interpretAsBlobDesc(graph->getOutputNodeByName(name)->getParentEdgesAtPort(0)[0]->getMemory());
            if (!data) {
                InferenceEngine::TensorDesc desc = _networkOutputs[name]->getTensorDesc();
                desc.setPrecision(normalizeToSupportedPrecision(desc.getPrecision()));

                // WA: need to avoid exception thrown when we compare blocking desc in SetBlob
                // in situation if we push output blobs as inputs for next network (in Hetero plugin)
                // it may be that output tensor desc will be different from real input tensor desc for next network
                // because the optimal descriptor was chosen (e.g. inPlace case for Split node)
                auto currBlockDesc = InferenceEngine::BlockingDesc(desc.getBlockingDesc().getBlockDims(), desc.getBlockingDesc().getOrder());
                desc = InferenceEngine::TensorDesc(desc.getPrecision(), desc.getDims(), currBlockDesc);

                data = make_blob_with_precision(desc);
                data->allocate();
            } else {
                const auto& expectedTensorDesc = pBlobDesc;

                if (expectedTensorDesc.getPrecision() != data->getTensorDesc().getPrecision()) {
                    IE_THROW(ParameterMismatch) << "Network input and output use the same name: " << name << " but expect blobs with different precision: "
                                                << data->getTensorDesc().getPrecision() << " for input and " << expectedTensorDesc.getPrecision()
                                                << " for output.";
                }

                if (expectedTensorDesc.getDims() != data->getTensorDesc().getDims()) {
                    IE_THROW(ParameterMismatch) << "Network input and output use the same name: " << name << " but expect blobs with different shapes.";
                }

                if (data->getTensorDesc().getLayout() != InferenceEngine::Layout::ANY && expectedTensorDesc.getLayout() != InferenceEngine::Layout::ANY &&
                    expectedTensorDesc.getBlockingDesc() != data->getTensorDesc().getBlockingDesc()) {
                    IE_THROW(ParameterMismatch) << "Network input and output use the same name: " << name
                                                << " but expect blobs with different blocking descriptors.";
                }
            }

            _outputs[name] = data;
            if (!externalPtr.count(name) && data->getTensorDesc() == pBlobDesc) {
                externalPtr[name] = data;
            }
        }
        data = _outputs[name];
        checkBlob(data, name, false);
    }
    if (!data) {
        IE_THROW() << "Cannot find blob with name: " << name;
    }
    return data;
}

void LegacyInferRequest::PushInputData() {
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

/* ========================================== InferRequest ========================================== */
InferRequest::InferRequest(const std::vector<std::shared_ptr<const ov::Node>>& inputs,
                           const std::vector<std::shared_ptr<const ov::Node>>& outputs,
                           ExecNetwork::Ptr execNetwork)
: InferRequestBase(inputs, outputs, execNetwork) {
    for (const std::shared_ptr<const ov::Node>& in : inputs) {
        modelInputsMap[ov::op::util::get_ie_output_name(ngraph::Output<const ngraph::Node>(in))] = in;
    }
    for (const std::shared_ptr<const ov::Node>& out : outputs) {
        modelOutputsMap[ov::op::util::get_ie_output_name(out->input_value(0))] = out;
    }

    CreateInferRequest();
}

void InferRequest::initBlobs() {
    for (const auto& it : modelInputsMap) {
        InferRequest::GetBlob(it.first);
    }
    for (const auto& it : modelOutputsMap) {
        InferRequest::GetBlob(it.first);
    }
}

void InferRequest::SetBlob(const std::string& name, const InferenceEngine::Blob::Ptr &data) {
    OV_ITT_SCOPED_TASK(itt::domains::intel_cpu, "SetBlob");
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
            IE_THROW() << "Can't set blob with name: " << name << ", because has null pointer to input node";
        }
        isInput = true;
    } else if (outputNodeItr != modelOutputsMap.end()) {
        if (!outputNodeItr->second) {
            IE_THROW() << "Can't set blob with name: " << name << ", because has null pointer to output node";
        }
        isInput = false;
    } else {
        IE_THROW(NotFound) << "Can't set blob with name: " << name << ", because input/output with this name doesn't exist";
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
            IE_THROW() << "Can't set input blob with name: " << name
                       << ", because model input (shape=" << shape
                       << ") and blob (shape=" << vec2str(data->getTensorDesc().getDims()) << ") are incompatible";
        }

        if (!isDynamic && ngraph::shape_size(shape.to_shape()) != data->size()) {
            IE_THROW() << "Can't set input blob with name: " << name << ", because model input size = " << ngraph::shape_size(shape.to_shape())
                       << " and blob size = " << data->size() << " are different.";
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
            graph->_normalizePreprocMap.find(name) == graph->_normalizePreprocMap.end()) {
            externalPtr[name] = data;
        } else if (externalPtr.find(name) != externalPtr.end()) {
            externalPtr.erase(name);
        }
        _inputs[name] = data;
        _batched_inputs.erase(name);
    } else {
        if (compoundBlobPassed) {
            IE_THROW(NotImplemented) << "Can't set compound blob: supported only for input pre-processing";
        }
        const auto netOutPrc = InferenceEngine::details::convertPrecision(outputNodeItr->second->get_input_element_type(0));
        if (netOutPrc != blobDesc.getPrecision()) {
            IE_THROW(ParameterMismatch) << "Failed to set input blob with precision: "
                               << blobDesc.getPrecision() << ", if CNNNetwork output blob precision is: " << netOutPrc;
        }

        const auto shape = outputNodeItr->second->get_input_partial_shape(0);
        const bool isDynamic = shape.is_dynamic();

        if (!shape.compatible(ov::PartialShape(data->getTensorDesc().getDims()))) {
            IE_THROW() << "Can't set output blob with name: " << name
                       << ", because model output (shape=" << shape
                       << ") and blob (shape=" << vec2str(data->getTensorDesc().getDims()) << ") are incompatible";
        }

        if (!isDynamic && ngraph::shape_size(shape.to_shape()) != data->size()) {
            IE_THROW() << "Can't set output blob with name: " << name << ", because model output size = " << ngraph::shape_size(shape.to_shape())
                       << " and blob size = " << data->size() << " are different.";
        }

        const auto &desc = graph->getOutputNodeByName(name)->getParentEdgesAtPort(0)[0]->getMemory().getDesc();
        if (!isDynamic && blobDesc == MemoryDescUtils::convertToTensorDesc(desc)) {
            externalPtr[name] = data;
        } else if (externalPtr.find(name) != externalPtr.end()) {
            externalPtr.erase(name);
        }
        _outputs[name] = data;
        outputControlBlocks.erase(name); // now the memory is under user's control
    }
}

void InferRequest::SetBlobsImpl(const std::string& name, const InferenceEngine::BatchedBlob::Ptr& batched_blob) {
    _batched_inputs[name] = batched_blob;
}

InferenceEngine::Blob::Ptr InferRequest::GetBlob(const std::string& name) {
    OV_ITT_SCOPED_TASK(itt::domains::intel_cpu, "GetBlob");

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
                    IE_THROW() << "Can't get blob with name: " << name << ", because has null pointer to input node";
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
                    graph->_normalizePreprocMap.find(name) == graph->_normalizePreprocMap.end()) {
                    externalPtr[name] = _inputs[name];
                }
            } else {
                IE_THROW() << "Blob with name: " << name << " exists in CPU plugin graph, but absents in network inputs";
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
                const auto& model_shape = outputNode->second->get_input_partial_shape(0);
                const auto& graph_shape = output->second->getInputShapeAtPort(0);

                // WA, due to the transformations and constant folding, shape inference of the resulting model may
                // have static shapes, while they are dynamic in the initial representation
                const auto& shape = graph_shape.isDynamic() ? model_shape :
                    (model_shape.is_dynamic() ? graph_shape.toPartialShape() : model_shape);

                const bool isDynamic = shape.is_dynamic();

                if (!data) {
                    InferenceEngine::SizeVector dims;
                    if (isDynamic) {
                        const auto model_prec = InferenceEngine::details::convertPrecision(outputNode->second->get_input_element_type(0));
                        const auto graph_prec = output->second->getParentEdgesAtPort(0)[0]->getMemory().getDesc().getPrecision();
                        OutputControlBlock control_block{model_prec, Shape{shape}};

                        DEBUG_LOG(name,
                            ", blob ", control_block.blob(),
                            ", tensor ", control_block.tensor(),
                            ", memmngr ", control_block.tensor()->get_memory()->getMemoryMngr(),
                            "memory object ", control_block.tensor()->get_memory().get());

                        data = control_block.blob();
                        if (model_prec == graph_prec) outputControlBlocks.emplace(std::make_pair(name, std::move(control_block)));
                    } else {
                        dims = shape.to_shape();

                        InferenceEngine::TensorDesc desc(InferenceEngine::details::convertPrecision(outputNode->second->get_input_element_type(0)),
                                                        dims, InferenceEngine::TensorDesc::getLayoutByRank(dims.size()));
                        data = make_blob_with_precision(desc);
                        data->allocate();
                    }
                } else {
                    const auto& blobDims = data->getTensorDesc().getDims();
                    // in static shape case is enough information that shapes are incompatible to throw exception
                    // but in dynamic shape case we also need to handle following corner case:
                    // on blob initialization stage we create empty blob with dimensions equal 0
                    // so if we have blob with all zero dimension we mustn't throw exception
                    if (!shape.compatible(ov::PartialShape(blobDims)) &&
                        (!isDynamic || static_cast<int64_t>(blobDims.size()) != shape.rank().get_length() ||
                         std::any_of(blobDims.begin(), blobDims.end(), [](const size_t& dims) {
                             return dims != 0;
                         }))) {
                        IE_THROW(ParameterMismatch) << "Network input and output use the same name: " << name
                                                    << ", but expect blobs with different shapes. Input shape: "
                                                    << ov::PartialShape(blobDims) << ", output shape: " << shape;
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
                    data->getTensorDesc() == MemoryDescUtils::convertToTensorDesc(output->second->getParentEdgesAtPort(0)[0]->getMemory().getDesc())) {
                    externalPtr[name] = data;
                }
            } else {
                IE_THROW() << "Blob with name: " << name << " exists in CPU plugin graph, but absents in network outputs";
            }
        }
        data = _outputs[name];
    }

    if (!data) {
        IE_THROW() << "Cannot find blob with name: " << name;
    }

    DEBUG_LOG(name, ", blob ", data, ", ", static_cast<void*>(data->buffer()));
    return data;
}

void InferRequest::checkBlobs() {
    for (auto const& input : _inputs) {
        checkBlob(input.second, input.first, true);
    }

    // won't check dynamic output blobs as they are not allocated.
    for (auto const& output : _outputs) {
        const auto out_node = findOutputByNodeName(output.first);
        const auto isDynamic = out_node && out_node->get_output_partial_shape(0).is_dynamic();
        if (!isDynamic) checkBlob(output.second, output.first, false);
    }
}

void InferRequest::PushInputData() {
    for (auto input : _inputs) {
        auto inputName = input.first;
        if (!modelInputsMap[inputName]) {
            IE_THROW() << "Input blobs map contains not registered during IInferencePlugin::LoadNetwork blob with name " << inputName;
        }

        pushInput(inputName, input.second, normToInputSupportedPrec(input));
    }
}

InferRequestBase::OutputControlBlock::OutputControlBlock(const InferenceEngine::Precision& precision, const Shape& shape) {
    dnnl::engine eng(dnnl::engine::kind::cpu, 0);
    m_buffers[m_buffIndx] = std::make_shared<MemoryMngrWithReuse>();
    m_proxyMemMngr = std::make_shared<ProxyMemoryMngr>(m_buffers[m_buffIndx]);

    Shape memShape = shape.isDynamic() ?
        Shape{VectorDims(shape.getRank(), 0)} : // this is a WA since the ITensor doesn't allow dyn shapes
        Shape{shape};

    CpuBlockedMemoryDescPtr desc =
        std::make_shared<CpuBlockedMemoryDesc>(precision, memShape);

    auto memory = std::make_shared<Memory>(eng, desc, m_proxyMemMngr);
    m_tensor = std::make_shared<Tensor>(memory);
    m_blob = tensor_to_blob({m_tensor, nullptr});
}

}   // namespace intel_cpu
}   // namespace ov
