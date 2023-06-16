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
#include "compiled_model.h"
#include "dnnl_extension_utils.h"
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
        auto port_name = get_port_name(in);
        _input_ports_map[port_name] = in;
    }
    for (const auto& out : get_outputs()) {
        auto port_name = get_port_name(out);
        _output_ports_map[port_name] = out;
    }

    // Sometimes tansfromation will change input/output name, such as SimpleIf test, need handle it.
    auto orig_model = _compiled_model->get_orig_model();
    for (const auto& in : orig_model->inputs()) {
        auto port_name = get_port_name(in);
        _orig_ports_map[port_name] = in;
        if (_input_ports_map.find(port_name) == _input_ports_map.end()) {
            for (auto& it : _input_ports_map) {
                auto& p = it.second;
                auto _name = p.get_node_shared_ptr()->get_friendly_name();
                if (port_name == _name) {
                    _input_ports_map[port_name] = p;
                    _port_name_change = true;
                }
            }
        }
    }
    for (const auto& out : orig_model->outputs()) {
        auto port_name = get_port_name(out);
        _orig_ports_map[port_name] = out;
        if (_output_ports_map.find(port_name) == _output_ports_map.end()) {
            for (auto& it : _output_ports_map) {
                auto& p = it.second;
                auto _name = p.get_node_shared_ptr()->get_friendly_name();
                if (port_name == _name) {
                    _output_ports_map[port_name] = p;
                    _port_name_change = true;
                }
            }
        }
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
        init_tensor(it.first);
    }
    for (const auto& it : _output_ports_map) {
        init_tensor(it.first);
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

    if (tensor.data() == nullptr) {
        OPENVINO_THROW("Input blob has no allocated memory");
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
        std::string name = query_port_name(port);
        if (name.empty()) {
            OPENVINO_THROW("compiled model doesn't contain this input port.");
        }
        const auto inputNode = cpuInputNodes.find(name);
        if (inputNode == cpuInputNodes.end())
            OPENVINO_THROW("CPU execution graph doesn't contain input node with name: ", name.c_str());
        if (inputNode->second->isDynamicNode()) {
            auto tensor = get_tensor(port);
            inputNode->second->redefineOutputMemory({tensor.get_shape()});
        }
    }
}

void SyncInferRequest::update_external_inputs() {
    // Update it due to batched_tensors case will update input tensor
    if (m_batched_tensors.size() == 0)
        return;
    for (auto input : get_inputs()) {
        std::string input_name = query_port_name(input);
        if (input_name.empty()) {
            OPENVINO_THROW("Input tensor map contains not registered during IPlugin::compile_model tensor with name ",
                           input_name);
        }
        if (_external_ptr.find(input_name) != _external_ptr.end()) {
            auto tensor = get_tensor(input);
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

    graph->PullOutputData(_outputs);
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

std::string SyncInferRequest::get_port_name(const ov::Output<const ov::Node>& port) const {
    bool is_input = ov::op::util::is_parameter(port.get_node());
    if (is_input) {
        return ov::op::util::get_ie_output_name(port);
    } else {
        const auto node = port.get_node_shared_ptr();
        return ov::op::util::get_ie_output_name(node->input_value(0));
    }
    return {};
}

std::string SyncInferRequest::query_port_name(const ov::Output<const ov::Node>& port) const {
    auto name = get_port_name(port);
    if (!_port_name_change)
        return name;

    // In some case, input name has been changed after transformation, but user maybe pass
    // orignal model's port, it need be aligned here.
    bool is_input = ov::op::util::is_parameter(port.get_node());
    if (is_input) {
        const auto& inMap = graph->inputNodesMap;
        if (inMap.find(name) == inMap.end()) {
            name = port.get_node_shared_ptr()->get_friendly_name();
        }
    } else {
        const auto& outMap = graph->outputNodesMap;
        if (outMap.find(name) == outMap.end()) {
            name = port.get_node_shared_ptr()->get_friendly_name();
        }
    }
    return name;
}

void SyncInferRequest::check_port(const ov::Output<const ov::Node>& port) const {
    auto name = get_port_name(port);
    if (name.empty()) {
        OPENVINO_THROW("cpu plugin checking port failed: cannot find this port with empty name.");
    }

    bool is_input = ov::op::util::is_parameter(port.get_node());
    if (is_input) {
        if (_input_ports_map.find(name) == _input_ports_map.end()) {
            OPENVINO_THROW("cpu plugin checking input port failed: cannot find this port with name ", name);
        }
    } else {
        if (_output_ports_map.find(name) == _output_ports_map.end()) {
            OPENVINO_THROW("cpu plugin checking output port failed: cannot find this port with name ", name);
        }
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

ov::Tensor SyncInferRequest::get_tensor(const ov::Output<const ov::Node>& _port) const {
    check_port(_port);
    auto port = get_internal_port(_port);
    return ov::ISyncInferRequest::get_tensor(port);
}

std::vector<ov::Tensor> SyncInferRequest::get_tensors(const ov::Output<const ov::Node>& _port) const {
    check_port(_port);
    auto port = get_internal_port(_port);
    return ov::ISyncInferRequest::get_tensors(port);
}

bool SyncInferRequest::check_precision_changed(const ov::Output<const ov::Node>& port) const {
    auto name = get_port_name(port);
    bool is_input = ov::op::util::is_parameter(port.get_node());

    if (is_input) {
        if (_input_ports_map.find(name) == _input_ports_map.end() ||
            _orig_ports_map.find(name) == _orig_ports_map.end()) {
            OPENVINO_THROW("cpu plugin checking input port's precision failed: cannot find this port!");
        }
        return _input_ports_map[name].get_element_type() != _orig_ports_map[name].get_element_type();
    } else {
        if (_output_ports_map.find(name) == _output_ports_map.end() ||
            _orig_ports_map.find(name) == _orig_ports_map.end()) {
            OPENVINO_THROW("cpu plugin checking output port's precision failed: cannot find this port!");
        }

        return _output_ports_map[name].get_element_type() != _orig_ports_map[name].get_element_type();
    }
    return false;
}

const ov::Output<const ov::Node>& SyncInferRequest::get_internal_port(const ov::Output<const ov::Node>& port) const {
    auto name = query_port_name(port);
    bool is_input = ov::op::util::is_parameter(port.get_node());
    if (is_input) {
        return _input_ports_map[name];
    } else {
        return _output_ports_map[name];
    }
}

ov::Tensor SyncInferRequest::create_internal_tensor(const ov::Tensor& tensor,
                                                    const ov::Output<const ov::Node>& port,
                                                    const std::string& name) {
    auto tensor_prec = InferenceEngine::details::convertPrecision(tensor.get_element_type());
    InferenceEngine::Precision inPrec = InferenceEngine::details::convertPrecision(port.get_element_type());
    bool needConvert = inPrec != tensor_prec;

    const void* srcData = tensor.data();
    if (srcData == nullptr) {
        OPENVINO_THROW("Input tensor has no allocated memory");
    }

    if (needConvert) {
        auto it = _internal_tensors.find(name);
        ov::Tensor new_tensor;
        if (it == _internal_tensors.end()) {
            new_tensor = ov::Tensor(port.get_element_type(), tensor.get_shape());
            _internal_tensors[name] = new_tensor;
        } else {
            new_tensor = it->second;
            if (new_tensor.get_element_type() != tensor.get_element_type()) {
                new_tensor = ov::Tensor(port.get_element_type(), tensor.get_shape());
                _internal_tensors[name] = new_tensor;
            }
            if (tensor.get_size() != new_tensor.get_size()) {
                new_tensor.set_shape(tensor.get_shape());
            }
        }

        if (tensor.get_size() != new_tensor.get_size()) {
            OPENVINO_THROW("Can't copy tensor: input and converted tensors have different number of elements: ",
                           tensor.get_size(),
                           " and ",
                           new_tensor.get_size());
        }

        void* dstData = new_tensor.data();
        if (dstData == nullptr) {
            OPENVINO_THROW("Converted input tensor has no allocated memory");
        }
        cpu_convert(srcData,
                    dstData,
                    tensor_prec,
                    InferenceEngine::details::convertPrecision(new_tensor.get_element_type()),
                    new_tensor.get_size());
        return new_tensor;
    }
    return tensor;
}

void SyncInferRequest::set_tensor(const ov::Output<const ov::Node>& _port, const ov::Tensor& _tensor) {
    OV_ITT_SCOPED_TASK(itt::domains::intel_cpu, "set_tensor");
    if (!_tensor)
        OPENVINO_THROW("Failed to set empty tensor for port!");
    check_port(_port);
    auto port = _port;
    auto tensor = _tensor;

    // WA: legacy api create blob with ANY layout will not set BlockingDesc, which will lead to tensor.get_shape()
    // return empty shape but tensor.get_size() will return correct value, and tensor.reshape() will not update
    // BlockingDesc, so have to construct new tensor with original tensor's data, which is only for ov legacy api usage.
    if (_port.get_partial_shape().is_static() && _tensor.get_size() > 0 && _tensor.get_shape().size() == 0 &&
        _tensor.get_size() == ov::shape_size(_port.get_shape())) {
        tensor = ov::Tensor(_tensor.get_element_type(), _port.get_shape(), _tensor.data());
    }

    // In case of import model, we cannot get original model info from the imported_model, so have to get it when
    // set_tensor
    auto name = get_port_name(_port);
    auto is_imported_model = _compiled_model->get_property(ov::loaded_from_cache.name()).as<bool>();
    if (is_imported_model) {
        _orig_ports_map[name] = _port;
    }

    auto precision_changed = check_precision_changed(_port);
    if (precision_changed) {
        auto _orig_port = _orig_ports_map[name];
        port = get_internal_port(_port);
        if ((_orig_port.get_element_type() != _tensor.get_element_type()) &&
            (port.get_element_type() != _tensor.get_element_type())) {
            IE_THROW(ParameterMismatch) << "Failed to set input tensor with precision: " << _tensor.get_element_type()
                                        << ", if model input tensor precision is: " << _port.get_element_type();
        }
        tensor = create_internal_tensor(_tensor, port, name);
    } else {
        if (_port.get_element_type() != _tensor.get_element_type()) {
            // Import model cannot get original port info if it is chained in meta plugin, need convert tensor here
            if (is_imported_model) {
                tensor = create_internal_tensor(_tensor, _port, name);
            } else {
                IE_THROW(ParameterMismatch)
                    << "Failed to set input tensor with precision: " << _tensor.get_element_type()
                    << ", if model input tensor precision is: " << _port.get_element_type();
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
            IE_THROW() << "Can't set output tensor with name: " << name << ", because model output (shape=" << shape
                       << ") and blob (shape=" << vec2str(tensor.get_shape()) << ") are incompatible";
        }

        if (!isDynamic && ngraph::shape_size(shape.to_shape()) != tensor.get_size()) {
            IE_THROW() << "Can't set output tensor with name: " << name
                       << ", because model output size = " << ngraph::shape_size(shape.to_shape())
                       << " and blob size = " << tensor.get_size() << " are different.";
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
        std::string input_name = query_port_name(input);
        if (input_name.empty()) {
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
