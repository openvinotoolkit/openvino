// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "infer_request.h"

#include <debug.h>
#include <ie_common.h>

#include <blob_factory.hpp>
#include <ie_ngraph_utils.hpp>
#include <map>
#include <string>
#include <transformations/utils/utils.hpp>
#include <vector>

#include "async_infer_request.h"
#include "dnnl_extension_utils.h"
#include "exec_network.h"
#include "itt.h"
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "memory_state.h"
#include "nodes/common/cpu_convert.h"
#include "nodes/common/cpu_memcpy.h"
#include "nodes/concat.h"
#include "nodes/memory.hpp"
#include "nodes/split.h"
#include "openvino/core/shape.hpp"
#include "openvino/runtime/tensor.hpp"
#include "utils/cpu_utils.hpp"
#include "utils/general_utils.h"

namespace ov {
namespace intel_cpu {

SyncInferRequest::SyncInferRequest(std::shared_ptr<const CompiledModel> compiled_model)
    : ov::ISyncInferRequest(compiled_model),
      _compiled_model(compiled_model) {
    for (const auto& in : get_inputs()) {
        // align with Graph::CreateGraph
        _input_ports_map[ov::op::util::get_ie_output_name(in)] = in;
    }

    for (const auto& out : get_outputs()) {
        // align with Graph::CreateGraph
        const auto node = out.get_node_shared_ptr();
        const std::string name = ov::op::util::get_ie_output_name(node->input_value(0));
        _output_ports_map[name] = out;
    }

    create_infer_request();
}

void SyncInferRequest::create_infer_request() {
    auto id = (_compiled_model->_numRequests)++;
    _profiling_task = openvino::itt::handle("INTEL_CPU_INFER_" + _compiled_model->_name + "_" + std::to_string(id));

    if (_compiled_model->_graphs.size() == 0)
        IE_THROW() << "No graph was found";
    graph = &(_compiled_model->GetGraph()._graph);

    // alocate memory for each tensor
    for (const auto& it : _input_ports_map) {
        prepare_tensor(it.first);
    }
    for (const auto& it : _output_ports_map) {
        prepare_tensor(it.first);
    }

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

            _memory_states.emplace_back(new VariableState(state_name, state_store));
        }
    }
}

SyncInferRequest::~SyncInferRequest() {
    --(_compiled_model->_numRequests);
}

void SyncInferRequest::pushInput(const std::string& inputName, ov::Tensor& tensor, InferenceEngine::Precision inPrec) {
    auto tensor_prec = InferenceEngine::details::convertPrecision(tensor.get_element_type());
    bool needConvert = inPrec != tensor_prec;

    const void* srcData = tensor.data();
    if (srcData == nullptr) {
        IE_THROW() << "Input blob has no allocated memory";
    }

    ov::Tensor iconv;
    if (needConvert) {
        iconv = ov::Tensor(tensor.get_element_type(), tensor.get_shape());
        if (tensor.get_size() != iconv.get_size())
            IE_THROW() << "Can't copy tensor: input and converted tensors have different number of elements: "
                       << tensor.get_size() << " and " << iconv.get_size();

        void* dstData = iconv.data();
        if (dstData == nullptr) {
            IE_THROW() << "Converted input tensor has no allocated memory";
        }
        cpu_convert(srcData,
                    dstData,
                    tensor_prec,
                    InferenceEngine::details::convertPrecision(iconv.get_element_type()),
                    iconv.get_size());
    }

    graph->PushInputData(inputName, needConvert ? iconv : tensor);
}

void SyncInferRequest::PushStates() {
    for (auto& node : graph->GetNodes()) {
        if (node->getType() == Type::MemoryInput) {
            auto cur_node = dynamic_cast<node::MemoryInput*>(node.get());
            if (!cur_node) {
                IE_THROW() << "Cannot cast " << node->getName() << " to MemoryInput";
            }
            auto cur_id = cur_node->getId();
            for (const auto& state : _memory_states) {
                if (state->get_name() == cur_id) {
                    auto cur_state_mem = cur_node->getStore();
                    auto data_ptr = state->get_state().data();
                    auto data_size = state->get_state().get_byte_size();
                    auto cur_state_mem_buf = static_cast<uint8_t*>(cur_state_mem->GetPtr());

                    cpu_memcpy(cur_state_mem_buf, data_ptr, data_size);
                }
            }
        }
    }
}

void SyncInferRequest::PullStates() {
    for (auto& node : graph->GetNodes()) {
        if (node->getType() == Type::MemoryInput) {
            auto cur_node = dynamic_cast<node::MemoryInput*>(node.get());
            if (!cur_node) {
                IE_THROW() << "Cannot cast " << node->getName() << " to MemoryInput";
            }
            auto cur_id = cur_node->getId();
            for (const auto& state : _memory_states) {
                if (state->get_name() == cur_id) {
                    auto cur_state_mem = cur_node->getStore();
                    auto data_ptr = state->get_state().data();
                    auto data_size = state->get_state().get_byte_size();
                    auto cur_state_mem_buf = static_cast<uint8_t*>(cur_state_mem->GetPtr());

                    cpu_memcpy(data_ptr, cur_state_mem_buf, data_size);
                }
            }
        }
    }
}

void SyncInferRequest::redefineMemoryForInputNodes() {
    const auto cpuInputNodes = graph->GetInputNodesMap();

    for (const auto& port : get_inputs()) {
        std::string name;
        if (!find_port_name(port, name)) {
            IE_THROW() << "compiled model doesn't contain this input port.";
        }
        const auto inputNode = cpuInputNodes.find(name);
        if (inputNode == cpuInputNodes.end())
            IE_THROW() << "CPU execution graph doesn't contain input node with name: " << name.c_str();
        if (inputNode->second->isDynamicNode()) {
            auto tensor = get_tensor(port);
            inputNode->second->redefineOutputMemory({tensor.get_shape()});
        }
    }
}

void SyncInferRequest::infer() {
    using namespace openvino::itt;
    OV_ITT_SCOPED_TASK(itt::domains::intel_cpu, _profiling_task);
    auto graphLock = _compiled_model->GetGraph();
    graph = &(graphLock._graph);

    throw_if_canceled();
    convert_batched_tensors();

    if (graph->hasDynamicInput()) {
        redefineMemoryForInputNodes();
    }

    changeDefaultPtr();

    throw_if_canceled();

    PushInputData();

    if (_memory_states.size() != 0) {
        PushStates();
    }

    graph->Infer(this);

    if (_memory_states.size() != 0) {
        PullStates();
    }

    throw_if_canceled();

    graph->PullOutputData(_outputs);
}

std::vector<ov::ProfilingInfo> SyncInferRequest::get_profiling_info() const {
    if (!graph || !graph->IsReady())
        IE_THROW() << "Graph is not ready!";
    std::vector<ov::ProfilingInfo> perfMap;
    graph->GetPerfData(perfMap);
    return perfMap;
}

static inline void changeEdgePtr(const EdgePtr& edge, void* newPtr) {
    edge->getMemoryPtr()->setDataHandle(newPtr);
}

void SyncInferRequest::changeDefaultPtr() {
    for (auto& it : _external_ptr) {
        const auto& inputNodesMap = graph->GetInputNodesMap();
        auto input = inputNodesMap.find(it.first);
        if (input != inputNodesMap.end()) {
            NodePtr inputNodePtr = input->second;
            if (inputNodePtr->getChildEdgeAt(0)->getMemory().GetData() == it.second)
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

                if (child->getType() == Type::Concatenation) {
                    auto concat = dynamic_cast<node::Concat*>(child.get());
                    if (concat && concat->isOptimized()) {
                        canBeInPlace = false;
                        break;
                    }
                }

                // Cannot be in-place before split because split is using different ptrs without offsets
                if (child->getType() == Type::Split) {
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

                    if (e->getMemory().GetData() == ce->getMemory().GetData()) {
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
            if (parentEdge->getMemory().GetData() == it.second)
                continue;

            bool canBeInPlace = true;
            void* defaultPtr = parentEdge->getMemory().GetData();
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

                    if (e->getMemory().GetData() == defaultPtr) {
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

std::vector<std::shared_ptr<ov::IVariableState>> SyncInferRequest::query_state() const {
    return _memory_states;
}

void SyncInferRequest::set_async_request(AsyncInferRequest* asyncRequest) {
    _asyncRequest = asyncRequest;
}

void SyncInferRequest::throw_if_canceled() const {
    if (_asyncRequest != nullptr) {
        _asyncRequest->throw_if_canceled();
    }
}

InferenceEngine::Precision SyncInferRequest::normToInputSupportedPrec(
    const std::pair<const std::string, ov::Tensor>& input) const {
    auto inPrec = InferenceEngine::details::convertPrecision(input.second.get_element_type());
    if (graph->hasMeanImageFor(input.first) &&
        one_of(inPrec, InferenceEngine::Precision::U8, InferenceEngine::Precision::BOOL)) {
        inPrec = InferenceEngine::Precision::FP32;
    } else {
        inPrec = normalizeToSupportedPrecision(inPrec);
    }

    if (inPrec == InferenceEngine::Precision::UNSPECIFIED) {
        IE_THROW() << "Unsupported input precision " << input.second.get_element_type();
    }

    return inPrec;
}

bool SyncInferRequest::find_port_name(const ov::Output<const ov::Node>& port, std::string& name, bool is_input) const {
    auto& map = is_input ? _input_ports_map : _output_ports_map;
    for (const auto& it : map) {
        if (it.second == port) {
            name = it.first;
            return true;
        }
    }
    return false;
}

void SyncInferRequest::set_tensor(const ov::Output<const ov::Node>& port, const ov::Tensor& tensor) {
    OV_ITT_SCOPED_TASK(itt::domains::intel_cpu, "set_tensor");

    std::string name;
    bool is_input = false;
    if (find_port_name(port, name, true)) {
        is_input = true;
    } else if (find_port_name(port, name, false)) {
        is_input = false;
    } else {
        IE_THROW(NotFound) << "Can't find port in input/output map";
    }

    if (name.empty()) {
        IE_THROW(NotFound) << "Can't set tensor with name: " << name
                           << ", because input/output with this name doesn't exist";
    }

    if (!tensor)
        IE_THROW(NotAllocated) << "Failed to set empty tensor with port name: \'" << name << "\'";

    InferenceEngine::TensorDesc tensor_desc(InferenceEngine::details::convertPrecision(tensor.get_element_type()),
                                            tensor.get_shape(),
                                            InferenceEngine::TensorDesc::getLayoutByRank(tensor.get_shape().size()));

    if (is_input) {
        const auto netInPrc = port.get_element_type();
        if (netInPrc != tensor.get_element_type()) {
            IE_THROW(ParameterMismatch) << "Failed to set input tensor with precision: " << tensor.get_element_type()
                                        << ", if model input tensor precision is: " << netInPrc;
        }

        const auto shape = port.get_partial_shape();
        const bool isDynamic = shape.is_dynamic();
        if (!shape.compatible(ov::PartialShape(tensor.get_shape()))) {
            IE_THROW() << "Can't set input tensor with name: " << name << ", because model input (shape=" << shape
                       << ") and tensor (shape=" << vec2str(tensor.get_shape()) << ") are incompatible";
        }

        if (!isDynamic && ngraph::shape_size(shape.to_shape()) != tensor.get_size()) {
            IE_THROW() << "Can't set input tensor with name: " << name
                       << ", because model input size = " << ngraph::shape_size(shape.to_shape())
                       << " and tensor size = " << tensor.get_size() << " are different.";
        }

        MemoryDescPtr actualDesc = graph->getInputNodeByName(name)->getBaseMemDescAtOutputPort(0);
        if (!actualDesc->isDefined()) {
            // we must define desc for dynamic case
            // otherwise we got incorrect check on shape compatibility inside isCompatible
            // because lower and upper bound will be compared
            actualDesc = actualDesc->cloneWithNewDims(tensor_desc.getLayout() == InferenceEngine::Layout::SCALAR
                                                          ? InferenceEngine::SizeVector{1}
                                                          : tensor_desc.getDims());
        }
        if (actualDesc->isCompatible(MemoryDescUtils::convertToCpuBlockedMemoryDesc(tensor_desc)) &&
            graph->_normalizePreprocMap.find(name) == graph->_normalizePreprocMap.end() &&
            !graph->getConfig().batchLimit) {
            _external_ptr[name] = tensor.data();
        } else if (_external_ptr.find(name) != _external_ptr.end()) {
            _external_ptr.erase(name);
        }
    } else {
        const auto netOutPrc = port.get_element_type();
        if (netOutPrc != tensor.get_element_type()) {
            IE_THROW(ParameterMismatch) << "Failed to set input tensor with precision: " << tensor.get_element_type()
                                        << ", if model output tensor precision is: " << netOutPrc;
        }

        const auto shape = port.get_partial_shape();
        const bool isDynamic = shape.is_dynamic();

        if (!shape.compatible(ov::PartialShape(tensor.get_shape()))) {
            IE_THROW() << "Can't set output tensor with name: " << name << ", because model output (shape=" << shape
                       << ") and blob (shape=" << vec2str(tensor.get_shape()) << ") are incompatible";
        }

        if (!isDynamic && ngraph::shape_size(shape.to_shape()) != tensor.get_size()) {
            IE_THROW() << "Can't set output tensor with name: " << name
                       << ", because model output size = " << ngraph::shape_size(shape.to_shape())
                       << " and blob size = " << tensor.get_size() << " are different.";
        }

        const auto& desc = graph->getOutputNodeByName(name)->getParentEdgesAtPort(0)[0]->getMemory().getDesc();
        if (!isDynamic && tensor_desc == MemoryDescUtils::convertToTensorDesc(desc) && !graph->getConfig().batchLimit) {
            _external_ptr[name] = tensor.data();
        } else if (_external_ptr.find(name) != _external_ptr.end()) {
            _external_ptr.erase(name);
        }
        _outputs[name] = tensor;
    }
    ov::ISyncInferRequest::set_tensor(port, tensor);
}

void SyncInferRequest::set_tensors_impl(const ov::Output<const ov::Node> port, const std::vector<ov::Tensor>& tensors) {
    auto found_port = find_port(port);

    if (found_port.is_input()) {
        m_batched_tensors.at(get_inputs().at(found_port.idx).get_tensor_ptr()) = tensors;
    } else {
        m_batched_tensors.at(get_outputs().at(found_port.idx).get_tensor_ptr()) = tensors;
    }
}

ov::Tensor SyncInferRequest::get_tensor(const ov::Output<const ov::Node>& port) const {
    return ov::ISyncInferRequest::get_tensor(port);
}

void SyncInferRequest::prepare_tensor(const std::string& name) {
    OV_ITT_SCOPED_TASK(itt::domains::intel_cpu, "prepare_tensor");

    if (!graph || !graph->IsReady())
        IE_THROW() << "Graph is not ready!";

    if (name.empty())
        IE_THROW() << "Can't preapre tensor for empty name! ";

    ov::Tensor tensor;
    const auto& inMap = graph->inputNodesMap;
    auto input = inMap.find(name);
    if (input != inMap.end()) {
        auto input_port = _input_ports_map.find(name);
        if (input_port != _input_ports_map.end()) {
            auto& port = input_port->second;
            tensor = ov::ISyncInferRequest::get_tensor(port);

            if (!tensor) {
                const auto shape = port.get_partial_shape();
                const bool isDynamic = shape.is_dynamic();
                ov::Shape tensor_shape;
                if (isDynamic) {
                    tensor_shape = ov::Shape(shape.rank().get_length(), 0);
                } else {
                    tensor_shape = shape.to_shape();
                }
                InferenceEngine::TensorDesc desc(InferenceEngine::details::convertPrecision(port.get_element_type()),
                                                 tensor_shape,
                                                 InferenceEngine::TensorDesc::getLayoutByRank(tensor_shape.size()));
                tensor = ov::Tensor(port.get_element_type(), tensor_shape);
                ov::ISyncInferRequest::set_tensor(port, tensor);

                if (!isDynamic &&
                    desc == MemoryDescUtils::convertToTensorDesc(
                                graph->getInputNodeByName(name)->getChildEdgesAtPort(0)[0]->getMemory().getDesc()) &&
                    graph->_normalizePreprocMap.find(name) == graph->_normalizePreprocMap.end() &&
                    !graph->getConfig().batchLimit) {
                    _external_ptr[name] = tensor.data();
                }
            }
        } else {
            IE_THROW() << "Tensor with name: " << name << " exists in CPU plugin graph, but absents in network inputs";
        }
    }

    const auto& outMap = graph->outputNodesMap;
    auto output = outMap.find(name);
    if (output != outMap.end()) {
        auto output_port = _output_ports_map.find(name);
        if (output_port != _output_ports_map.end()) {
            auto& port = output_port->second;
            tensor = ov::ISyncInferRequest::get_tensor(port);
            const auto shape = port.get_partial_shape();
            const bool isDynamic = shape.is_dynamic();

            if (!tensor) {
                ov::Shape tensor_shape;
                if (isDynamic) {
                    tensor_shape = ov::Shape(shape.rank().get_length(), 0);
                } else {
                    tensor_shape = shape.to_shape();
                }
                tensor = ov::Tensor(port.get_element_type(), tensor_shape);
                ov::ISyncInferRequest::set_tensor(port, tensor);
            } else {
                const auto& blobDims = tensor.get_shape();
                // in static shape case is enough information that shapes are incompatible to throw exception
                // but in dynamic shape case we also need to handle following corner case:
                // on blob initialization stage we create empty blob with dimensions equal 0
                // so if we have blob with all zero dimension we mustn't throw exception
                if (!shape.compatible(ov::PartialShape(blobDims)) &&
                    (!isDynamic || blobDims.size() != shape.rank().get_length() ||
                     std::any_of(blobDims.begin(), blobDims.end(), [](const size_t& dims) {
                         return dims != 0;
                     }))) {
                    IE_THROW(ParameterMismatch)
                        << "Network input and output use the same name: " << name
                        << ", but expect tensors with different shapes. Input shape: " << ov::PartialShape(blobDims)
                        << ", output shape: " << shape;
                }

                const auto netOutPrc = port.get_element_type();
                if (netOutPrc != tensor.get_element_type()) {
                    IE_THROW(ParameterMismatch)
                        << "Network input and output use the same name: " << name
                        << " but expect blobs with different precision: " << tensor.get_element_type()
                        << " for input and " << netOutPrc << " for output.";
                }
            }
            _outputs[name] = tensor;
            InferenceEngine::TensorDesc desc(InferenceEngine::details::convertPrecision(tensor.get_element_type()),
                                             tensor.get_shape(),
                                             InferenceEngine::TensorDesc::getLayoutByRank(tensor.get_shape().size()));
            if (!isDynamic && !_external_ptr.count(name) &&
                desc == MemoryDescUtils::convertToTensorDesc(
                            output->second->getParentEdgesAtPort(0)[0]->getMemory().getDesc()) &&
                !graph->getConfig().batchLimit) {
                _external_ptr[name] = tensor.data();
            }
        } else {
            IE_THROW() << "Tensor with name: " << name << " exists in CPU plugin graph, but absents in network outputs";
        }
    }

    if (!tensor) {
        IE_THROW() << "Cannot find tensor with name: " << name;
    }

    return;
}

void SyncInferRequest::PushInputData() {
    for (auto input : get_inputs()) {
        std::string input_name;
        if (!find_port_name(input, input_name)) {
            IE_THROW() << "Input tensor map contains not registered during IPlugin::compile_model tensor with name "
                       << input_name;
        }

        auto tensor = get_tensor(input);
        std::pair<const std::string, ov::Tensor> in(input_name, tensor);
        pushInput(input_name, tensor, normToInputSupportedPrec(in));
    }
}

}  // namespace intel_cpu
}  // namespace ov
