// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "infer_request.h"

#include "async_infer_request.h"
#include "dnnl_extension_utils.h"
#include "itt.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "nodes/common/cpu_convert.h"
#include "nodes/memory_state_base.h"
#include "openvino/core/shape.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/runtime/threading/cpu_message.hpp"
#include "proxy_mem_blk.h"
#include "utils/general_utils.h"
#include "utils/ngraph_utils.hpp"

using OvString = ov::element_type_traits<ov::element::string>::value_type;

namespace ov::intel_cpu {
SyncInferRequest::SyncInferRequest(CompiledModelHolder compiled_model)
    : ov::ISyncInferRequest(compiled_model.compiled_model()),
      m_compiled_model(std::move(compiled_model)) {
    const auto& inputs = get_inputs();
    for (std::size_t input_index = 0; input_index < inputs.size(); input_index++) {
        m_input_ports_map[input_index] = inputs[input_index];
    }

    const auto& outputs = get_outputs();
    for (std::size_t output_index = 0; output_index < outputs.size(); output_index++) {
        m_output_ports_map[output_index] = outputs[output_index];
    }
    create_infer_request();
}

void SyncInferRequest::create_infer_request() {
    m_profiling_task = openvino::itt::handle("INTEL_CPU_INFER_" + m_compiled_model.name() + "_" +
                                             std::to_string(m_compiled_model.id()));

    // Alocate memory for each tensor if static shape
    for (const auto& it : m_input_ports_map) {
        init_tensor(it.first, ov::ISyncInferRequest::FoundPort::Type::INPUT);
    }
    for (const auto& it : m_output_ports_map) {
        init_tensor(it.first, ov::ISyncInferRequest::FoundPort::Type::OUTPUT);
    }

    // create states according to the list of the MemoryStateNodes
    m_memory_states = m_compiled_model.graph().memoryStates();
}

void SyncInferRequest::redefine_memory_for_input_nodes(Graph& graph) {
    for (const auto& input_port : m_input_ports_map) {
        auto inputNode = graph.getInputNodeByIndex(input_port.first);
        OPENVINO_ASSERT(inputNode, "CPU execution graph doesn't contain output node with index: ", input_port.first);
        if (inputNode->isDynamicNode()) {
            auto tensor = get_tensor(input_port.second);
            inputNode->redefineOutputMemory({tensor->get_shape()});
        }
    }
}

void SyncInferRequest::update_external_tensor_ptrs() {
    // Update it due to batched_tensors case will update input tensor
    for (const auto& input : m_input_ports_map) {
        if (m_input_external_ptr.find(input.first) != m_input_external_ptr.end()) {
            auto tensor = get_tensor(input.second);
            m_input_external_ptr[input.first] = tensor;
        }
    }
}

void SyncInferRequest::infer() {
    using namespace openvino::itt;
    OV_ITT_SCOPED_TASK(itt::domains::intel_cpu, m_profiling_task);
    auto graphLock = m_compiled_model.lock();
    auto&& graph = graphLock._graph;
    auto message = ov::threading::message_manager();

    throw_if_canceled();
    if (m_asyncRequest->m_has_sub_infers) {
        sub_streams_infer();
        message->server_wait();
        return;
    }

    convert_batched_tensors();
    if (m_batched_tensors.size() > 0) {
        // batched_tensors will be updated for each infer, external_ptr should be update together
        update_external_tensor_ptrs();
    }

    if (graph.hasDynamicInput()) {
        redefine_memory_for_input_nodes(graph);
    }

    change_default_ptr(graph);

    throw_if_canceled();

    // state -> node
    if (!m_memory_states.empty()) {
        graph.assignStates(m_memory_states);
    }

    push_input_data(graph);

    graph.Infer(this);

    throw_if_canceled();

    // update output control blocks, if any, in order to refresh internal buffers
    if (graph.IsDynamic()) {
        for (auto&& item : m_outputControlBlocks) {
            item.second.update();
        }
    }

    graph.PullOutputData(m_outputs);
}

std::vector<ov::ProfilingInfo> SyncInferRequest::get_profiling_info() const {
    auto&& graph = m_compiled_model.graph();
    if (!graph.IsReady()) {
        OPENVINO_THROW("Graph is not ready!");
    }
    std::vector<ov::ProfilingInfo> perfMap;
    graph.GetPerfData(perfMap);
    return perfMap;
}

static inline void change_edge_ptr(const EdgePtr& edge, ov::SoPtr<ov::ITensor>& tensor) {
    auto mem = edge->getMemoryPtr();
    OPENVINO_ASSERT(mem != nullptr, "Edge with name '", *edge, "' doesn't have allocated memory object.");

    if (tensor->get_element_type() == element::string) {
        auto memBlock = dynamic_cast<StringMemory*>(mem.get())->getStringMemoryBlockPtr();
        OPENVINO_ASSERT(memBlock);
        memBlock->setExtBuff(tensor->data<StringMemory::OvString>(), tensor->get_size());
    } else {
        auto memBlock = mem->getMemoryBlock();
        OPENVINO_ASSERT(memBlock);
        memBlock->setExtBuff(tensor->data(), tensor->get_byte_size());
    }
}

void SyncInferRequest::change_default_ptr(Graph& graph) {
    std::unordered_set<const void*> inputPtrs;
    std::function<void(const EdgePtr& edge, ov::SoPtr<ov::ITensor>& tensor)> changeInpPtr;
    if (graph.IsDynamic()) {
        changeInpPtr = [&inputPtrs](const EdgePtr& edge, ov::SoPtr<ov::ITensor>& tensor) {
            change_edge_ptr(edge, tensor);
            inputPtrs.insert(tensor->data());
        };
    } else {
        changeInpPtr = [](const EdgePtr& edge, ov::SoPtr<ov::ITensor>& tensor) {
            change_edge_ptr(edge, tensor);
        };
    }

    for (auto& it : m_input_external_ptr) {
        auto inputNodePtr = graph.getInputNodeByIndex(it.first);
        OPENVINO_ASSERT(inputNodePtr, "Cannot find input tensor with index: ", it.first);
        if (inputNodePtr->getDstDataAtPort(0) == static_cast<void*>(it.second->data())) {
            continue;
        }
        auto& childEdges = inputNodePtr->getChildEdges();
        // Perform checks that the user's memory will not be modified
        bool canBeInPlace = true;
        for (auto& childEdge : childEdges) {
            auto ce = childEdge.lock();
            if (!ce) {
                OPENVINO_THROW("Node ", inputNodePtr->getName(), " contains empty child edge");
            }

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
                if (!e) {
                    OPENVINO_THROW("Node ", inputNodePtr->getName(), " contains empty child edge");
                }
                changeInpPtr(e, it.second);
            }
        }
    }

    for (auto& it : m_output_external_ptr) {
        auto output = graph.getOutputNodeByIndex(it.first);
        OPENVINO_ASSERT(output, "Cannot find output tensor with index: ", it.first);
        auto parentEdge = output->getParentEdgeAt(0);
        void* const outputRawPtr = parentEdge->getMemory().getData();
        if (outputRawPtr == static_cast<void*>(it.second->data())) {
            continue;
        }

        bool canBeInPlace = true;
        // Cannot be in-place after concat because concat is using different ptrs without offsets
        auto parent = parentEdge->getParent();
        NodePtr previousParent;
        auto parent_port = parentEdge->getInputNum();
        do {
            previousParent = parent;
            if (parent->getChildEdgesAtPort(parent_port).size() != 1 || parent->isConstant()) {
                canBeInPlace = false;
                break;
            }
            if (parent->getChildEdgeAt(parent_port)->inPlace(Edge::LOOK_UP)) {
                canBeInPlace = false;
                break;
            }

            auto& parentEdges = parent->getParentEdges();
            for (auto& edge : parentEdges) {
                auto e = edge.lock();
                if (!e) {
                    OPENVINO_THROW("Node ", parent->getName(), " contains empty parent edge");
                }

                if (parent_port == parent->inPlaceInputPort(e->getOutputNum())) {
                    parent = e->getParent();
                    parent_port = e->getInputNum();
                    break;
                }
            }
        } while (previousParent != parent);
        if (canBeInPlace) {
            change_edge_ptr(parentEdge, it.second);
        }
    }

    if (graph.IsDynamic()) {
        const auto& outMemBlocksMap = graph.getOutputNodesMemBlocksMap();
        for (auto&& item : outMemBlocksMap) {
            const auto index = item.first;

            // share intel_cpu::Tensor to Graph by injecting to corresponding ProxyMemoryBlock instance.
            auto outputMemBlock = item.second;
            OPENVINO_ASSERT(outputMemBlock, "proxy mem block for output ", index, " is empty.");

            auto controlBlockItr = m_outputControlBlocks.find(index);

            if (controlBlockItr != m_outputControlBlocks.end()) {
                auto output = graph.getOutputNodeByIndex(index);
                OPENVINO_ASSERT(output, "Output with index: ", index, " is absent in the outputNodesMap");
                auto parentEdge = output->getParentEdgeAt(0);
                // avoid cyclic memory use
                auto&& controlBlock = controlBlockItr->second;

                std::shared_ptr<IMemoryBlock> memBlock =
                    inputPtrs.count(controlBlock.rawPtr()) ?  // same memory is used on the input and output
                        controlBlock.nextMemBlock()
                                                           :  // then swap internal buffer to avoid data corruption
                        controlBlock.currentMemBlock();       // else reuse the existing buffer

                outputMemBlock->setMemBlockResize(std::move(memBlock));
                DEBUG_LOG("reset proxy ",
                          outputMemBlock,
                          ", actual ",
                          controlBlock.currentMemBlock(),
                          " graph ",
                          &graph,
                          " infer request ",
                          this);
                DEBUG_LOG(index, ", tensor ", controlBlock.tensor());
            } else {
                outputMemBlock->reset();  // switch to the internal memory since memory sharing is no longer possible
            }
        }
    }
}

std::vector<ov::SoPtr<ov::IVariableState>> SyncInferRequest::query_state() const {
    if (m_asyncRequest->m_has_sub_infers) {
        auto requests = m_asyncRequest->getSubInferRequest();
        std::vector<ov::SoPtr<ov::IVariableState>> states;
        for (const auto& request : requests) {
            auto cur = request->query_state();
            states.insert(states.end(), cur.begin(), cur.end());
        }
        return states;
    }
    return {m_memory_states.begin(), m_memory_states.end()};
}

void SyncInferRequest::set_async_request(AsyncInferRequest* asyncRequest) {
    m_asyncRequest = asyncRequest;
}

void SyncInferRequest::throw_if_canceled() const {
    if (m_asyncRequest != nullptr) {
        m_asyncRequest->throw_if_canceled();
    }
}

ov::SoPtr<ov::ITensor> SyncInferRequest::get_tensor(const ov::Output<const ov::Node>& in_port) const {
    auto port = get_internal_port(in_port);
    return ov::ISyncInferRequest::get_tensor(port);
}

std::vector<ov::SoPtr<ov::ITensor>> SyncInferRequest::get_tensors(const ov::Output<const ov::Node>& in_port) const {
    auto port = get_internal_port(in_port);
    return ov::ISyncInferRequest::get_tensors(port);
}

const ov::Output<const ov::Node>& SyncInferRequest::get_internal_port(const ov::Output<const ov::Node>& port) const {
    auto port_find = find_port(port);
    OPENVINO_ASSERT(port_find.found(), "Can not find port: ", port.get_any_name());
    if (port_find.is_input()) {
        return m_input_ports_map.at(port_find.idx);
    }
    return m_output_ports_map.at(port_find.idx);
}

void SyncInferRequest::set_tensor(const ov::Output<const ov::Node>& in_port, const ov::SoPtr<ov::ITensor>& in_tensor) {
    OV_ITT_SCOPED_TASK(itt::domains::intel_cpu, "set_tensor");
    if (!in_tensor) {
        OPENVINO_THROW("Failed to set empty tensor for port!");
    }
    auto port = get_internal_port(in_port);
    auto tensor = in_tensor;

    // WA: legacy api create blob with ANY layout will not set BlockingDesc, which will lead to tensor.get_shape()
    // return empty shape but tensor.get_size() return correct value, and tensor.reshape() cannot update
    // BlockingDesc, so to construct new tensor with original tensor's data, which is only for ov legacy api usage.
    if (in_port.get_partial_shape().is_static() && in_tensor->get_size() > 0 && in_tensor->get_shape().size() == 0 &&
        in_tensor->get_size() == ov::shape_size(in_port.get_shape()) && in_port.get_shape().size() > 0) {
        tensor = ov::make_tensor(in_tensor->get_element_type(), in_port.get_shape(), in_tensor->data());
    }
    auto port_found = find_port(in_port);
    auto mem_desc_ptr = MemoryDescUtils::generateCpuBlockedMemoryDesc(tensor);
    if (port_found.is_input()) {
        auto input_index = port_found.idx;
        const auto netInPrc = port.get_element_type();
        if (netInPrc != tensor->get_element_type()) {
            OPENVINO_THROW("ParameterMismatch: Failed to set tensor for input with precision: ",
                           tensor->get_element_type(),
                           ", since the model input tensor precision is: ",
                           netInPrc);
        }

        const auto& shape = port.get_partial_shape();
        const bool isDynamic = shape.is_dynamic();
        if (!shape.compatible(ov::PartialShape(tensor->get_shape()))) {
            OPENVINO_THROW("Can't set the input tensor with index: ",
                           input_index,
                           ", because the model input (shape=",
                           shape,
                           ") and the tensor (shape=",
                           vec2str(tensor->get_shape()),
                           ") are incompatible");
        }

        if (!isDynamic && ov::shape_size(shape.to_shape()) != tensor->get_size()) {
            OPENVINO_THROW("Can't set input tensor with index: ",
                           input_index,
                           ", because the model input size = ",
                           ov::shape_size(shape.to_shape()),
                           " and the tensor size = ",
                           tensor->get_size(),
                           " are different.");
        }

        auto&& graph = m_compiled_model.graph();

        auto inputNode = graph.getInputNodeByIndex(input_index);
        OPENVINO_ASSERT(inputNode, "CPU execution graph doesn't contain input node with index: ", input_index);

        MemoryDescPtr actualDesc = inputNode->getBaseMemDescAtOutputPort(0);
        if (!actualDesc->isDefined()) {
            // we must define desc for dynamic case
            // otherwise we got incorrect check on shape compatibility inside isCompatible
            // because lower and upper bound will be compared
            actualDesc = actualDesc->cloneWithNewDims(
                ov::is_scalar(tensor->get_shape()) ? VectorDims{1} : VectorDims{tensor->get_shape()});
        }

        if (actualDesc->isCompatible(*mem_desc_ptr)) {
            m_input_external_ptr[input_index] = tensor;
        } else if (m_input_external_ptr.find(input_index) != m_input_external_ptr.end()) {
            m_input_external_ptr.erase(input_index);
        }
    } else {
        auto output_index = port_found.idx;
        const auto netOutPrc = port.get_element_type();
        if (netOutPrc != tensor->get_element_type()) {
            OPENVINO_THROW("ParameterMismatch: Failed to set tensor for output with precision: ",
                           tensor->get_element_type(),
                           ", if model output tensor precision is: ",
                           netOutPrc);
        }

        const auto& shape = port.get_partial_shape();
        const bool isDynamic = shape.is_dynamic();

        if (!shape.compatible(ov::PartialShape(tensor->get_shape())) && tensor->get_size() != 0) {
            OPENVINO_THROW("Can't set the output tensor with index: ",
                           output_index,
                           ", because the model output tensor (shape=",
                           shape,
                           ") and the current tensor (shape=",
                           vec2str(tensor->get_shape()),
                           ") are incompatible");
        }

        if (!isDynamic && ov::shape_size(shape.to_shape()) != tensor->get_size()) {
            OPENVINO_THROW("Can't set the output tensor with index: ",
                           output_index,
                           ", because the model output size = ",
                           ov::shape_size(shape.to_shape()),
                           " and the currernt tensor size = ",
                           tensor->get_size(),
                           " are different.");
        }

        auto&& graph = m_compiled_model.graph();

        auto outputNode = graph.getOutputNodeByIndex(output_index);
        OPENVINO_ASSERT(outputNode, "CPU execution graph doesn't contain output node with index: ", output_index);
        const auto& desc = outputNode->getParentEdgeAt(0)->getMemory().getDesc();
        if (!isDynamic && mem_desc_ptr->isCompatible(desc)) {
            m_output_external_ptr[output_index] = tensor;
        } else if (m_output_external_ptr.find(output_index) != m_output_external_ptr.end()) {
            m_output_external_ptr.erase(output_index);
        }

        m_outputs[output_index] = tensor;
        m_outputControlBlocks.erase(output_index);  // now the memory is under user's control
    }
    ov::ISyncInferRequest::set_tensor(port, tensor);
}

void SyncInferRequest::set_tensors_impl(const ov::Output<const ov::Node> port,
                                        const std::vector<ov::SoPtr<ITensor>>& tensors) {
    if (find_port(port).is_input()) {
        m_batched_tensors[port.get_tensor_ptr()] = tensors;
        return;
    }
    OPENVINO_THROW("Cannot find port to set_tensors!");
}

void SyncInferRequest::init_tensor(const std::size_t& port_index, const ov::ISyncInferRequest::FoundPort::Type& type) {
    OV_ITT_SCOPED_TASK(itt::domains::intel_cpu, "init_tensor");
    auto&& graph = m_compiled_model.graph();
    OPENVINO_ASSERT(graph.IsReady(), "Graph is not ready!");

    ov::SoPtr<ITensor> tensor;
    if (type == ov::ISyncInferRequest::FoundPort::Type::INPUT) {
        OPENVINO_ASSERT(graph.getInputNodeByIndex(port_index),
                        "Tensor with index: ",
                        port_index,
                        " absent in the plugin's graph inputs");
        const auto& port = m_input_ports_map[port_index];
        tensor = ov::ISyncInferRequest::get_tensor(port);

        if (!tensor) {
            const auto& shape = port.get_partial_shape();
            const bool isDynamic = shape.is_dynamic();
            ov::Shape tensor_shape;
            if (isDynamic) {
                for (auto&& item : shape) {
                    tensor_shape.push_back(item.is_static() ? item.get_length() : 0);
                }
            } else {
                tensor_shape = shape.to_shape();
            }

            tensor = ov::make_tensor(port.get_element_type(), tensor_shape);
            ov::ISyncInferRequest::set_tensor(port, tensor);

            if (!isDynamic) {
                auto mem_desc_ptr = MemoryDescUtils::generateCpuBlockedMemoryDesc(tensor);
                auto inputNode = graph.getInputNodeByIndex(port_index);
                OPENVINO_ASSERT(inputNode, "CPU execution graph doesn't contain input node with index: ", port_index);
                if (mem_desc_ptr->isCompatible(inputNode->getChildEdgeAt(0)->getMemory().getDesc())) {
                    m_input_external_ptr[port_index] = tensor;
                }
            }
        }
    }

    if (type == ov::ISyncInferRequest::FoundPort::Type::OUTPUT) {
        auto output = graph.getOutputNodeByIndex(port_index);
        OPENVINO_ASSERT(output, "Tensor with index: ", port_index, " absent in the plugin's graph outputs");
        if (m_outputs.find(port_index) == m_outputs.end()) {
            const auto& port = m_output_ports_map[port_index];
            const auto& port_shape = port.get_partial_shape();
            const auto& graph_shape = output->getInputShapeAtPort(0);

            // WA, due to the transformations and constant folding, shape inference of the resulting model may
            // have static shapes, while they are dynamic in the initial representation
            const auto& shape = graph_shape.isDynamic()
                                    ? port_shape
                                    : (port_shape.is_dynamic() ? graph_shape.toPartialShape() : port_shape);

            const bool isDynamic = shape.is_dynamic();
            tensor = ov::ISyncInferRequest::get_tensor(port);

            if (!tensor) {
                ov::Shape tensor_shape;
                const auto model_prec = port.get_element_type();
                if (isDynamic) {
                    if (model_prec == element::string) {
                        VectorDims memDims;
                        auto c_shape = Shape{shape};
                        for (auto&& dim : c_shape.getDims()) {
                            memDims.push_back(dim != Shape::UNDEFINED_DIM ? dim : 0);
                        }

                        dnnl::engine eng(dnnl::engine::kind::cpu, 0);
                        CpuBlockedMemoryDescPtr desc =
                            std::make_shared<CpuBlockedMemoryDesc>(model_prec, Shape{memDims});
                        auto memory = std::make_shared<StringMemory>(eng, desc);

                        tensor = std::make_shared<Tensor>(memory);
                    } else {
                        const auto graph_prec = output->getParentEdgeAt(0)->getMemory().getDesc().getPrecision();
                        OutputControlBlock control_block{model_prec, Shape{shape}};

                        DEBUG_LOG(port_index,
                                  ", tensor ",
                                  control_block.tensor(),
                                  ", memBlock ",
                                  control_block.tensor()->get_memory()->getMemoryBlock(),
                                  "memory object ",
                                  control_block.tensor()->get_memory().get());

                        tensor = control_block.tensor();
                        if (model_prec == graph_prec) {
                            m_outputControlBlocks.emplace(port_index, std::move(control_block));
                        }
                    }
                } else {
                    tensor_shape = shape.to_shape();
                    tensor = ov::make_tensor(model_prec, tensor_shape);
                }
                ov::ISyncInferRequest::set_tensor(port, tensor);
            }
            m_outputs[port_index] = tensor;
            if (!port_shape.is_dynamic() && !m_output_external_ptr.count(port_index)) {
                auto desc = MemoryDescUtils::generateCpuBlockedMemoryDesc(tensor);
                if (desc->isCompatible(output->getParentEdgeAt(0)->getMemory().getDesc())) {
                    m_output_external_ptr[port_index] = tensor;
                }
            }
            // update tensors in case of multiple output ports with the same name
            for (const auto& out : m_output_ports_map) {
                if ((out.first == port_index) && tensor) {
                    ov::ISyncInferRequest::set_tensor(out.second, tensor);
                }
            }
        }
    }
    if (!tensor) {
        OPENVINO_THROW("Cannot find tensor with index: ", port_index);
    }
    return;
}

void SyncInferRequest::push_input_data(Graph& graph) {
    for (auto& input : m_input_ports_map) {
        auto tensor = get_tensor(input.second);
        graph.PushInputData(input.first, tensor);
    }
}

SyncInferRequest::OutputControlBlock::OutputControlBlock(const ov::element::Type& precision, const Shape& shape) {
    dnnl::engine eng(dnnl::engine::kind::cpu, 0);
    m_buffers[m_buffIndx] = std::make_shared<MemoryBlockWithReuse>();
    m_proxyMemBlock = std::make_shared<ProxyMemoryBlock>(m_buffers[m_buffIndx]);

    VectorDims memDims;
    if (shape.isDynamic()) {  // this is a WA since the ITensor doesn't allow dyn shapes
        for (auto&& item : shape.getDims()) {
            memDims.push_back(item != Shape::UNDEFINED_DIM ? item : 0);
        }
    } else {
        memDims = shape.getStaticDims();
    }

    CpuBlockedMemoryDescPtr desc = std::make_shared<CpuBlockedMemoryDesc>(precision, Shape{memDims});

    auto memory = std::make_shared<Memory>(eng, desc, m_proxyMemBlock);
    m_tensor = std::make_shared<Tensor>(memory);
}

void SyncInferRequest::sub_streams_infer() {
    std::map<ov::Output<const ov::Node>, ov::SoPtr<ov::ITensor>> input_tensors;
    auto message = ov::threading::message_manager();
    auto requests = m_asyncRequest->getSubInferRequest();
    auto inputs = get_inputs();
    auto outputs = get_outputs();

    size_t requests_num = requests.size();

    if (requests.size() > 0) {
        for (const auto& output : outputs) {
            auto tensor = requests[0]->get_tensor(output);
            set_tensor(output, tensor);
        }
        for (size_t i = 0; i < requests_num; i++) {
            for (auto& input : inputs) {
                auto tensor = get_tensor(input);
                requests[i]->set_tensor(input, tensor);
            }

            requests[i]->set_callback([message](const std::exception_ptr& ptr) {
                ov::threading::MessageInfo msg_info;
                msg_info.msg_type = ov::threading::MsgType::CALL_BACK;
                message->send_message(msg_info);
            });
        }
        for (size_t i = 0; i < requests_num; i++) {
            requests[i]->start_async();
        }
    }
}

}  // namespace ov::intel_cpu
