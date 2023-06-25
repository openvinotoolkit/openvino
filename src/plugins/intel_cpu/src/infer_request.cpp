// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "infer_request.h"

#include <map>
#include <string>
#include <vector>

#include "async_infer_request.h"
#include "blob_factory.hpp"
#include "compiled_model.h"
#include "debug.h"
#include "dnnl_extension_utils.h"
#include "ie_common.h"
#include "ie_ngraph_utils.hpp"
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
#include "transformations/utils/utils.hpp"
#include "utils/cpu_utils.hpp"
#include "utils/general_utils.h"

namespace ov {
namespace intel_cpu {
SyncInferRequest::SyncInferRequest(std::shared_ptr<const CompiledModel> compiled_model)
    : ov::ISyncInferRequest(compiled_model),
      _compiled_model(compiled_model) {
    _is_legacy_api = _compiled_model->GetGraph()._graph.getConfig().isLegacyApi;
    for (const auto& in : get_inputs()) {
        auto port_name = get_port_name(in, _is_legacy_api);
        _input_ports_map[port_name] = in;
    }
    for (const auto& out : get_outputs()) {
        auto port_name = get_port_name(out, _is_legacy_api);
        _output_ports_map[port_name] = out;
    }

    auto orig_model = _compiled_model->get_orig_model();
    for (const auto& in : orig_model->inputs()) {
        auto port_name = get_port_name(in, _is_legacy_api);
        _orig_ports_map[port_name] = in;
        if (_input_ports_map.find(port_name) == _input_ports_map.end()) {
            OPENVINO_THROW("Input port's name has been changed, cannot find ", port_name);
        }
        _port_precision_changed[port_name] =
            _input_ports_map[port_name].get_element_type() != _orig_ports_map[port_name].get_element_type();
    }
    for (const auto& out : orig_model->outputs()) {
        auto port_name = get_port_name(out, _is_legacy_api);
        _orig_ports_map[port_name] = out;
        if (_output_ports_map.find(port_name) == _output_ports_map.end()) {
            OPENVINO_THROW("Output port's name has been changed, cannot find ", port_name);
        }
        _port_precision_changed[port_name] =
            _output_ports_map[port_name].get_element_type() != _orig_ports_map[port_name].get_element_type();
    }
    create_infer_request();
}

void SyncInferRequest::create_infer_request() {
    auto id = (_compiled_model->_numRequests)++;
    _profiling_task = openvino::itt::handle("INTEL_CPU_INFER_" + _compiled_model->_name + "_" + std::to_string(id));

    if (_compiled_model->_graphs.size() == 0)
        OPENVINO_THROW("No graph was found");
    graph = &(_compiled_model->GetGraph()._graph);

    // alocate memory for each tensor if static shape
    for (const auto& it : _input_ports_map) {
        init_tensor(it.first);
    }
    for (const auto& it : _output_ports_map) {
        init_tensor(it.first);
        // allocate aux tensor for output if output precision has been changed
        get_tensor(it.second);
    }

    // Save all MemoryLayer data tensors. Will use insight about mechanics
    // of MemoryLayer implementation. It uses output edge of MemoryLayer
    // producer as storage for tensor to keep it between infer calls.
    for (auto& node : graph->GetNodes()) {
        if (node->getType() == Type::MemoryInput) {
            auto memoryNode = dynamic_cast<node::MemoryInput*>(node.get());
            if (!memoryNode) {
                OPENVINO_THROW("Cannot cast ", node->getName(), " to MemoryInput");
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

    if (tensor.data() == nullptr) {
        OPENVINO_THROW("Input tensor has no allocated memory");
    }

    if (needConvert) {
        OPENVINO_THROW("Tensor precision ", tensor_prec, " is mismatch with input precision ", inPrec);
    }

    graph->PushInputData(inputName, tensor);
}

void SyncInferRequest::PushStates() {
    for (auto& node : graph->GetNodes()) {
        if (node->getType() == Type::MemoryInput) {
            auto cur_node = dynamic_cast<node::MemoryInput*>(node.get());
            if (!cur_node) {
                OPENVINO_THROW("Cannot cast ", node->getName(), " to MemoryInput");
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
                OPENVINO_THROW("Cannot cast ", node->getName(), " to MemoryInput");
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
        std::string name = get_port_name(port, _is_legacy_api);
        if (name.empty()) {
            OPENVINO_THROW("compiled model doesn't contain this input port.");
        }
        const auto inputNode = cpuInputNodes.find(name);
        if (inputNode == cpuInputNodes.end())
            OPENVINO_THROW("CPU execution graph doesn't contain input node with name: ", name.c_str());
        if (inputNode->second->isDynamicNode()) {
            auto tensor = get_compiled_tensor(port);
            inputNode->second->redefineOutputMemory({tensor.get_shape()});
        }
    }
}

void SyncInferRequest::update_external_inputs() {
    // Update it due to batched_tensors case will update input tensor
    if (m_batched_tensors.size() == 0)
        return;
    for (auto input : get_inputs()) {
        std::string input_name = get_port_name(input, _is_legacy_api);
        if (input_name.empty()) {
            OPENVINO_THROW("Input tensor map contains not registered during IPlugin::compile_model tensor with name ",
                           input_name);
        }
        if (_external_ptr.find(input_name) != _external_ptr.end()) {
            auto tensor = get_compiled_tensor(input);
            _external_ptr[input_name] = tensor.data();
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
    update_external_inputs();

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

    graph->PullOutputData(_outputs, _port_precision_changed, _aux_tensors);
}

std::vector<ov::ProfilingInfo> SyncInferRequest::get_profiling_info() const {
    if (!graph || !graph->IsReady())
        OPENVINO_THROW("Graph is not ready!");
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
                    OPENVINO_THROW("Node ", inputNodePtr->getName(), " contains empty child edge");

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
                        OPENVINO_THROW("Node ", child->getName(), " contains empty child edge");

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
                        OPENVINO_THROW("Node ", inputNodePtr->getName(), " contains empty child edge");

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
                        OPENVINO_THROW("Node ", parent->getName(), " contains empty parent edge");

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
        OPENVINO_THROW("Cannot find input/output blob: ", it.first);
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
        OPENVINO_THROW("Unsupported input precision ", input.second.get_element_type());
    }

    return inPrec;
}

bool SyncInferRequest::check_compiled_port(const ov::Output<const ov::Node>& port) const {
    auto name = get_port_name(port, _is_legacy_api);
    if (name.empty()) {
        OPENVINO_THROW("cpu plugin checking port failed: cannot find this port with empty name.");
    }

    bool is_input = ov::op::util::is_parameter(port.get_node());
    if (is_input) {
        auto it = _input_ports_map.find(name);
        if (it == _input_ports_map.end()) {
            OPENVINO_THROW("cpu plugin checking input port failed: cannot find this port with name ", name);
        }

        if ((it->second.get_element_type() == port.get_element_type()) &&
            (it->second.get_partial_shape() == port.get_partial_shape())) {
            return true;
        }
        return false;
    } else {
        auto it = _output_ports_map.find(name);
        if (it == _output_ports_map.end()) {
            OPENVINO_THROW("cpu plugin checking output port failed: cannot find this port with name ", name);
        }
        if ((it->second.get_element_type() == port.get_element_type()) &&
            (it->second.get_partial_shape() == port.get_partial_shape())) {
            return true;
        }
        return false;
    }
}

InferenceEngine::TensorDesc SyncInferRequest::create_tensor_desc(const ov::Tensor& tensor) {
    auto element_type = tensor.get_element_type();
    auto shape = tensor.get_shape();
    std::vector<size_t> blk_order(shape.size());
    std::iota(blk_order.begin(), blk_order.end(), 0);
    std::vector<size_t> dim_offset(shape.size(), 0);
    std::vector<size_t> blk_strides;
    auto byte_strides = element_type.bitwidth() >= 8 ? tensor.get_strides() : Strides{};
    if (byte_strides.empty()) {
        blk_strides = ov::row_major_strides(shape);
    } else {
        blk_strides.resize(byte_strides.size());
        std::transform(byte_strides.begin(),
                       byte_strides.end(),
                       blk_strides.begin(),
                       [&element_type](size_t byte_stride) {
                           OPENVINO_ASSERT(byte_stride % element_type.size() == 0,
                                           "Limitation: Stride in bytes ",
                                           byte_stride,
                                           " should be divisible by size of element ",
                                           element_type.size());
                           return byte_stride / element_type.size();
                       });
    }
    return ie::TensorDesc{ie::details::convertPrecision(element_type),
                          shape,
                          ie::BlockingDesc{shape, blk_order, 0, dim_offset, blk_strides}};
}

ov::Tensor SyncInferRequest::get_compiled_tensor(const ov::Output<const ov::Node>& _port) const {
    check_compiled_port(_port);
    auto port = get_compiled_port(_port);
    return ov::ISyncInferRequest::get_tensor(port);
}

ov::Tensor SyncInferRequest::get_tensor(const ov::Output<const ov::Node>& _port) const {
    auto port_name = get_port_name(_port, _is_legacy_api);
    auto port = get_compiled_port(_port);
    auto compiled_tensor = ov::ISyncInferRequest::get_tensor(port);

    // No precision change
    auto is_precision_changed = _port_precision_changed[port_name];
    if (!is_precision_changed)
        return compiled_tensor;

    // If precision has been changed, it need return original precision tensor
    // port's data will be stored in _aux_tensors, and need converted to compiled tensor
    //     input  tensor: will be copied to compiled tensor before do graph inference
    //     output tensor: has be copied from graph's memory to aux tensor

    if (_orig_ports_map.find(port_name) == _orig_ports_map.end()) {
        OPENVINO_THROW("Cannot find original port, name: ", port_name);
    }

    // Find aux tensor, will create one if cannot find
    auto port_shape = port.get_partial_shape();
    auto it = _aux_tensors.find(port_name);
    ov::Shape aux_shape = compiled_tensor.get_shape();
    if (it == _aux_tensors.end()) {
        _aux_tensors[port_name] = ov::Tensor(_orig_ports_map[port_name].get_element_type(), aux_shape);
    } else if (port_shape.is_dynamic()) {
        if (_aux_tensors[port_name].get_shape() != aux_shape)
            _aux_tensors[port_name].set_shape(aux_shape);
    }

    return _aux_tensors[port_name];
}

std::vector<ov::Tensor> SyncInferRequest::get_tensors(const ov::Output<const ov::Node>& _port) const {
    // TODO: support set_tensors() for port with precision/shape changes
    auto port = get_compiled_port(_port);
    return ov::ISyncInferRequest::get_tensors(port);
}

const ov::Output<const ov::Node>& SyncInferRequest::get_compiled_port(const ov::Output<const ov::Node>& port) const {
    auto name = get_port_name(port, _is_legacy_api);
    bool is_input = ov::op::util::is_parameter(port.get_node());
    if (is_input) {
        return _input_ports_map[name];
    } else {
        return _output_ports_map[name];
    }
}

void SyncInferRequest::set_tensor(const ov::Output<const ov::Node>& _port, const ov::Tensor& _tensor) {
    OV_ITT_SCOPED_TASK(itt::domains::intel_cpu, "set_tensor");
    if (!_tensor)
        OPENVINO_THROW("Failed to set empty tensor for port!");
    auto is_compiled_port = check_compiled_port(_port);
    auto port = get_compiled_port(_port);
    auto tensor = _tensor;

    // WA: legacy api create blob with ANY layout will not set BlockingDesc, which will lead to tensor.get_shape()
    // return empty shape but tensor.get_size() return correct value, and tensor.reshape() cannot update
    // BlockingDesc, so to construct new tensor with original tensor's data, which is only for ov legacy api usage.
    if (_port.get_partial_shape().is_static() && _tensor.get_size() > 0 && _tensor.get_shape().size() == 0 &&
        _tensor.get_size() == ov::shape_size(_port.get_shape()) && _port.get_shape().size() > 0) {
        tensor = ov::Tensor(_tensor.get_element_type(), _port.get_shape(), _tensor.data());
    }
    auto name = get_port_name(_port, _is_legacy_api);
    auto is_precision_changed = _port_precision_changed[name];

    // Precision has been changed
    if (is_precision_changed) {
        if (!is_compiled_port) {
            // Orig port
            auto _orig_port = _orig_ports_map[name];
            if (_orig_port.get_element_type() == _tensor.get_element_type()) {
                // Orig port + orig port's tensor
                _aux_tensors[name] = _tensor;
                tensor = ov::ISyncInferRequest::get_tensor(port);
                tensor.set_shape(_tensor.get_shape());
            } else if (port.get_element_type() == _tensor.get_element_type()) {
                // Orig port + compiled port's tensor
                tensor = _tensor;
            } else {
                OPENVINO_THROW("Failed to set input tensor with precision: ",
                               _tensor.get_element_type(), ", if model input tensor precision is: ",
                               port.get_element_type(),
                               " or ",
                               _orig_port.get_element_type());
            }
        } else {
            // Compiled port
            if (_port.get_element_type() != _tensor.get_element_type()) {
                if (_orig_ports_map[name].get_element_type() == _tensor.get_element_type()) {
                    // origina_port precision tensor
                    _aux_tensors[name] = _tensor;
                    tensor = ov::ISyncInferRequest::get_tensor(port);
                    tensor.set_shape(_tensor.get_shape());
                } else {
                    IE_THROW(ParameterMismatch)
                        << "Failed to set input tensor with precision: " << _tensor.get_element_type()
                        << ", if model input tensor precision is: " << _port.get_element_type();
                }
            }
        }
    }

    auto tensor_desc = create_tensor_desc(tensor);
    bool is_input = ov::op::util::is_parameter(port.get_node());
    if (is_input) {
        const auto netInPrc = port.get_element_type();
        if (netInPrc != tensor.get_element_type()) {
            IE_THROW(ParameterMismatch) << "Failed to set input tensor with precision: " << tensor.get_element_type()
                                        << ", if model input tensor precision is: " << netInPrc;
        }

        const auto shape = port.get_partial_shape();
        const bool isDynamic = shape.is_dynamic();
        if (!shape.compatible(ov::PartialShape(tensor.get_shape()))) {
            OPENVINO_THROW("Can't set input tensor with name: ",
                           name,
                           ", because model input (shape=",
                           shape,
                           ") and tensor (shape=",
                           vec2str(tensor.get_shape()),
                           ") are incompatible");
        }

        if (!isDynamic && ngraph::shape_size(shape.to_shape()) != tensor.get_size()) {
            OPENVINO_THROW("Can't set input tensor with name: ",
                           name,
                           ", because model input size = ",
                           ngraph::shape_size(shape.to_shape()),
                           " and tensor size = ",
                           tensor.get_size(),
                           " are different.");
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
            graph->_normalizePreprocMap.find(name) == graph->_normalizePreprocMap.end()) {
            _external_ptr[name] = tensor.data();
        } else if (_external_ptr.find(name) != _external_ptr.end()) {
            _external_ptr.erase(name);
        }
    } else {
        const auto netOutPrc = port.get_element_type();
        if (netOutPrc != tensor.get_element_type()) {
            IE_THROW(ParameterMismatch) << "Failed to set output tensor with precision: " << tensor.get_element_type()
                                        << ", if model output tensor precision is: " << netOutPrc;
        }

        const auto shape = port.get_partial_shape();
        const bool isDynamic = shape.is_dynamic();

        if (!shape.compatible(ov::PartialShape(tensor.get_shape()))) {
            OPENVINO_THROW("Can't set output tensor with name: ",
                           name,
                           ", because model output (shape=",
                           shape,
                           ") and blob (shape=",
                           vec2str(tensor.get_shape()),
                           ") are incompatible");
        }

        if (!isDynamic && ngraph::shape_size(shape.to_shape()) != tensor.get_size()) {
            OPENVINO_THROW("Can't set output tensor with name: ",
                           name,
                           ", because model output size = ",
                           ngraph::shape_size(shape.to_shape()),
                           " and blob size = ",
                           tensor.get_size(),
                           " are different.");
        }

        const auto& desc = graph->getOutputNodeByName(name)->getParentEdgesAtPort(0)[0]->getMemory().getDesc();
        if (!isDynamic && tensor_desc == MemoryDescUtils::convertToTensorDesc(desc)) {
            _external_ptr[name] = tensor.data();
        } else if (_external_ptr.find(name) != _external_ptr.end()) {
            _external_ptr.erase(name);
        }
        _outputs[name] = tensor;
    }
    ov::ISyncInferRequest::set_tensor(port, tensor);
}

void SyncInferRequest::set_tensors_impl(const ov::Output<const ov::Node> port, const std::vector<ov::Tensor>& tensors) {
    for (const auto& input : get_inputs()) {
        if (input == port) {
            m_batched_tensors[input.get_tensor_ptr()] = tensors;
            return;
        }
    }
    OPENVINO_THROW("Cannot find port to set_tensors!");
}

void SyncInferRequest::init_tensor(const std::string& name) {
    OV_ITT_SCOPED_TASK(itt::domains::intel_cpu, "init_tensor");

    if (!graph || !graph->IsReady())
        OPENVINO_THROW("Graph is not ready!");

    if (name.empty())
        OPENVINO_THROW("Can't preapre tensor for empty name! ");

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

                tensor = ov::Tensor(port.get_element_type(), tensor_shape);
                ov::ISyncInferRequest::set_tensor(port, tensor);

                auto desc = create_tensor_desc(tensor);
                if (!isDynamic &&
                    desc == MemoryDescUtils::convertToTensorDesc(
                                graph->getInputNodeByName(name)->getChildEdgesAtPort(0)[0]->getMemory().getDesc()) &&
                    graph->_normalizePreprocMap.find(name) == graph->_normalizePreprocMap.end()) {
                    _external_ptr[name] = tensor.data();
                }
            }
        } else {
            OPENVINO_THROW("Tensor with name: ", name, " exists in CPU plugin graph, but absents in network inputs");
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
                    (!isDynamic || static_cast<int64_t>(blobDims.size()) != shape.rank().get_length() ||
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
            auto desc = create_tensor_desc(tensor);
            if (!isDynamic && !_external_ptr.count(name) &&
                desc == MemoryDescUtils::convertToTensorDesc(
                            output->second->getParentEdgesAtPort(0)[0]->getMemory().getDesc())) {
                _external_ptr[name] = tensor.data();
            }
        } else {
            OPENVINO_THROW("Tensor with name: ", name, " exists in CPU plugin graph, but absents in network outputs");
        }
    }

    if (!tensor) {
        OPENVINO_THROW("Cannot find tensor with name: ", name);
    }
    return;
}

void SyncInferRequest::PushInputData() {
    for (auto input : get_inputs()) {
        std::string input_name = get_port_name(input, _is_legacy_api);
        if (input_name.empty()) {
            OPENVINO_THROW("Input tensor map contains not registered during IPlugin::compile_model tensor with name ",
                           input_name);
        }
        auto tensor = get_compiled_tensor(input);
        if (_aux_tensors.find(input_name) != _aux_tensors.end()) {
            auto& aux_tensor = _aux_tensors[input_name];

            if (aux_tensor.get_shape() != tensor.get_shape()) {
                tensor.set_shape(aux_tensor.get_shape());
            }
            const void* srcData = aux_tensor.data();
            void* dstData = tensor.data();
            if ((dstData == nullptr) || (srcData == nullptr)) {
                OPENVINO_THROW("Get tensor has no allocated memory");
            }
            cpu_convert(srcData,
                        dstData,
                        InferenceEngine::details::convertPrecision(aux_tensor.get_element_type()),
                        InferenceEngine::details::convertPrecision(tensor.get_element_type()),
                        tensor.get_size());
        }
        std::pair<const std::string, ov::Tensor> in(input_name, tensor);
        pushInput(input_name, tensor, normToInputSupportedPrec(in));
    }
}
}  // namespace intel_cpu
}  // namespace ov
