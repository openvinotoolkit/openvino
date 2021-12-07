// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_tensoriterator_node.h"

#include <string>
#include <vector>
#include <mkldnn_extension_utils.h>
#include <ie_ngraph_utils.hpp>
#include <utils/general_utils.h>
#include "common/blocked_desc_creator.h"
#include "utils/ngraph_utils.hpp"
#include "transformations/utils/utils.hpp"

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine::details;

namespace MKLDNNPlugin {

#define THROW_ERROR IE_THROW() << getTypeStr() << " layer with name '" << getName() << "' "

static NodeConfig make_plain_config(const std::shared_ptr<ov::Node>& op) {
    NodeConfig config;

    for (size_t i = 0; i < op->get_input_size(); i++) {
        const auto &origShape = op->get_input_partial_shape(i);
        const auto& shape = Shape(origShape.rank().get_length() == 0 ? ov::PartialShape{1} : origShape);
        const auto prec = InferenceEngine::details::convertPrecision(op->get_input_element_type(i));

        PortConfig data_conf {};
        auto descCreator = BlockedDescCreator::getCommonCreators().at(LayoutType::ncsp);
        data_conf.desc = descCreator->createSharedDesc(prec, shape);
        config.inConfs.push_back(data_conf);
    }

    for (size_t i = 0; i < op->get_output_size(); i++) {
        const auto &origShape = op->get_output_partial_shape(i);
        const auto& shape = Shape(origShape.rank().get_length() == 0 ? ov::PartialShape{1} : origShape);
        const auto prec = InferenceEngine::details::convertPrecision(op->get_output_element_type(i));

        PortConfig data_conf {};
        auto descCreator = BlockedDescCreator::getCommonCreators().at(LayoutType::ncsp);
        data_conf.desc = descCreator->createSharedDesc(prec, shape);
        config.outConfs.push_back(data_conf);
    }

    config.dynBatchSupport = true;
    return config;
}

class PortIteratorHelper : public PortMapHelper {
public:
    PortIteratorHelper(const MKLDNNMemoryPtr &from, const MKLDNNMemoryPtr &to, bool sliced_src,
                       const PortMap &slice_rule, const mkldnn::engine& eng)
                       : m_sliced_src(sliced_src) {
        const auto &full_blob = sliced_src ? from : to;
        const auto &part_blob = !sliced_src ? from : to;

        auto axis = slice_rule.m_axis;
        auto stride = slice_rule.m_stride;

        auto full_dims = full_blob->GetShape().getStaticDims();
        auto part_dims = part_blob->GetShape().getStaticDims();

        auto abs_stride = std::abs(stride);
        auto sign_of_stride = stride < 0.0f ? -1 : 1;

        m_iter_count = full_dims[axis] / abs_stride;

        full_dims[axis] = abs_stride;
        IE_ASSERT(full_dims == part_dims) << "Shape mismatch for tensor iterator port";

        // make chunk view
        auto chunk_desc = full_blob->GetDescWithType<DnnlMemoryDesc>()->getDnnlDesc();
        chunk_desc.data.dims[axis] = abs_stride;
        chunk_desc.data.padded_dims[axis] = abs_stride;  // TODO: asamption that plain tensor

        m_full_mem = full_blob->GetPrimitive();
        const auto full_mem_handler = m_full_mem.get_data_handle();
        mkldnn::memory chunk_mem = {chunk_desc, eng, full_mem_handler};

        auto elem_size = MKLDNNExtensionUtils::sizeOfDataType(mkldnn::memory::data_type(chunk_desc.data.data_type));

        m_chunk_stride_in_byte = chunk_desc.data.format_desc.blocking.strides[axis] * elem_size * abs_stride;
        m_chunk_offset_in_byte = sign_of_stride < 0 ? (m_iter_count - 1) * m_chunk_stride_in_byte : 0;
        m_chunk_stride_in_byte *= sign_of_stride;

        if (sliced_src) {
            m_mem_holder_src = chunk_mem;
            m_mem_holder_dst = to->GetPrimitive();
        } else {
            m_mem_holder_src = from->GetPrimitive();
            m_mem_holder_dst = chunk_mem;
        }
        m_reorder = {m_mem_holder_src, m_mem_holder_dst};
    }

    void execute(mkldnn::stream strm, int iter) override {
        IE_ASSERT(iter >= 0 && iter < m_iter_count);

        auto &chunk_mem = m_sliced_src ? m_mem_holder_src : m_mem_holder_dst;
        chunk_mem.set_data_handle(static_cast<uint8_t *>(m_full_mem.get_data_handle()) +
                                          m_chunk_offset_in_byte + m_chunk_stride_in_byte * iter);

        m_reorder.execute(strm, m_mem_holder_src, m_mem_holder_dst);
    }

private:
    ptrdiff_t m_chunk_stride_in_byte = 0;
    ptrdiff_t m_chunk_offset_in_byte = 0;

    bool m_sliced_src;
    mkldnn::memory m_full_mem;

    int m_iter_count;
};

class BackEdgePortHelper : public PortMapHelper {
public:
    BackEdgePortHelper(const MKLDNNMemoryPtr &from, const MKLDNNMemoryPtr &to, const mkldnn::engine& eng) {
        m_mem_holder_src = from->GetPrimitive();
        m_mem_holder_dst = to->GetPrimitive();
        m_reorder = {m_mem_holder_src, m_mem_holder_dst};
    }

    void execute(mkldnn::stream strm, int iter) override {
        if (iter != 0) {
            m_reorder.execute(strm, m_mem_holder_src, m_mem_holder_dst);
        }
    }
};

class IterCountPortHelper : public PortMapHelper {
public:
    IterCountPortHelper(const MKLDNNMemoryPtr &to, const mkldnn::engine& eng) {
        // Only scalar I32 tensor is supported
        IE_ASSERT(to->GetDataType() == memory::data_type::s32);
        IE_ASSERT(to->GetShape() == Shape(VectorDims{1}));
        m_mem_holder_dst = to->GetPrimitive();
    }

    void execute(mkldnn::stream strm, int m_n_iter) override {
        auto mem = m_mem_holder_dst;
        auto data_ptr = static_cast<uint32_t*>(mem.get_data_handle());
        if (data_ptr == nullptr) {
            IE_THROW() << "TensorIterator node has not allocated memory for IterCountPortHelper";
        }
        *data_ptr = m_n_iter;
    }
};

class asBoolCheck : public PortChecker {
public:
    asBoolCheck(const MKLDNNMemoryPtr &mem) {
        IE_ASSERT(mem->GetDataType() == memory::data_type::u8);
        IE_ASSERT(mem->GetShape() == Shape(InferenceEngine::SizeVector{1}));
        m_mem_holder = mem->GetPrimitive();
    }

    int getStatus() override {
        auto data_ptr = static_cast<uint8_t*>(m_mem_holder.get_data_handle());
        if (data_ptr == nullptr) {
            IE_THROW() << "TensorIterator node has not allocated memory for asBoolCheck";
        }
        return *data_ptr == static_cast<uint8_t>(0) ? 0 : 1;
    }
};

class asIntCheck : public PortChecker {
public:
    asIntCheck(const MKLDNNMemoryPtr &mem) {
        IE_ASSERT(mem->GetDataType() == memory::data_type::s32);
        IE_ASSERT(mem->GetShape() == Shape(InferenceEngine::SizeVector{1}));
        m_mem_holder = mem->GetPrimitive();
    }

    int getStatus() override {
        auto data_ptr = static_cast<uint32_t*>(m_mem_holder.get_data_handle());
        if (data_ptr == nullptr) {
            IE_THROW() << "TensorIterator node has not allocated memory for asIntCheck";
        }
        return *data_ptr;
    }
};

class staticValueCheck : public PortChecker {
public:
    staticValueCheck(const int &value) : value(value) {}

    int getStatus() override {
        return value;
    }
private:
    int value;
};

DynamicBuffer::DynamicBuffer(const MKLDNNMemoryPtr &from, const MKLDNNMemoryPtr &to, const PortMap &map_rule) : m_from(from), m_to(to), m_map_rule(map_rule) {
    m_elem_size = MKLDNNExtensionUtils::sizeOfDataType(from->GetDataType());
}

void DynamicBuffer::execute(mkldnn::stream strm, const int iter) {
    if (iter == 0) {
        init(strm);
        return;
    }

    overwrite(strm);
}

void DynamicBuffer::init(mkldnn::stream strm) {
    m_chunk_offset_in_byte = 0;
    m_buffer_offset_in_byte = 0;

    auto eng = strm.get_engine();
    auto src_mem = m_from->GetPrimitive();
    auto src_desc = src_mem.get_desc();
    m_mem_holder_buffer.reset(new memory(src_desc, eng));
    copy(strm, src_mem, *m_mem_holder_buffer.get());
}

void DynamicBuffer::copy(mkldnn::stream strm, mkldnn::memory& src, mkldnn::memory& dst) {
    mkldnn::reorder reorder(src, dst);
    reorder.execute(strm, src, dst);
}

void DynamicBuffer::overwrite(mkldnn::stream strm) {
    auto eng = strm.get_engine();
    auto new_buffer = create_buffer(eng);
    move_buffer(strm, new_buffer);
    move_data(strm);
}

std::shared_ptr<mkldnn::memory> DynamicBuffer::create_buffer(const mkldnn::engine& eng) {
    const auto axis = m_map_rule.m_axis;
    const auto stride = m_map_rule.m_stride;
    const auto abs_stride = std::abs(stride);

    const auto old_desc = m_mem_holder_buffer->get_desc();
    auto dims = old_desc.dims();
    dims[axis] += m_from->getStaticDims()[axis];
    mkldnn::memory::desc new_buffer_desc(dims, old_desc.data_type(), MKLDNNExtensionUtils::GetPlainFormatByRank(dims.size()));

    if (stride > 0.0f) {
        m_chunk_offset_in_byte += new_buffer_desc.data.format_desc.blocking.strides[axis] * m_elem_size * abs_stride;
    } else {
        m_buffer_offset_in_byte = m_from->GetPrimitive().get_desc().data.format_desc.blocking.strides[axis] * m_elem_size * abs_stride;
    }

    return std::make_shared<mkldnn::memory>(new_buffer_desc, eng);
}

void DynamicBuffer::move_buffer(mkldnn::stream strm, std::shared_ptr<mkldnn::memory> new_buffer) {
    auto eng = strm.get_engine();
    auto axis = m_map_rule.m_axis;

    // make chunk view
    auto chunk_desc = new_buffer->get_desc();
    auto old_desc = m_mem_holder_buffer->get_desc();
    chunk_desc.data.dims[axis] = old_desc.data.dims[axis];
    chunk_desc.data.padded_dims[axis] = old_desc.data.padded_dims[axis];

    const auto new_mem_handler = new_buffer->get_data_handle();
    mkldnn::memory chunk_mem(chunk_desc, eng, new_mem_handler);
    chunk_mem.set_data_handle(static_cast<uint8_t *>(new_buffer->get_data_handle()) + m_buffer_offset_in_byte);

    copy(strm, *m_mem_holder_buffer.get(), chunk_mem);
    m_mem_holder_buffer = new_buffer;
}

void DynamicBuffer::move_data(mkldnn::stream strm) {
    auto eng = strm.get_engine();
    auto axis = m_map_rule.m_axis;
    auto src_mem = m_from->GetPrimitive();
    auto src_desc = src_mem.get_desc();

    // make chunk view
    auto chunk_desc = m_mem_holder_buffer->get_desc();
    chunk_desc.data.dims[axis] = src_desc.data.dims[axis];
    chunk_desc.data.padded_dims[axis] = src_desc.data.padded_dims[axis];

    const auto new_mem_handler = m_mem_holder_buffer->get_data_handle();
    mkldnn::memory chunk_mem = { chunk_desc, eng, new_mem_handler };
    chunk_mem.set_data_handle(static_cast<uint8_t*>(m_mem_holder_buffer->get_data_handle()) + m_chunk_offset_in_byte);

    copy(strm, src_mem, chunk_mem);
}

void DynamicBuffer::transfer(mkldnn::stream strm, MKLDNNNode* node) {
    const auto &currDesc = m_to->getDesc();
    if (!currDesc.getShape().isStatic() || currDesc.getShape().getStaticDims() != m_from->getStaticDims()) {
        const auto memDesc = node->getBaseMemDescAtOutputPort(m_map_rule.m_from)->cloneWithNewDims(
                MKLDNNExtensionUtils::convertToVectorDims(m_mem_holder_buffer->get_desc().dims()));
        m_to->redefineDesc(*memDesc);
    }

    auto dst_mem = m_to->GetPrimitive();
    copy(strm, *m_mem_holder_buffer.get(), dst_mem);
}

}  // namespace MKLDNNPlugin

bool MKLDNNTensorIteratorNode::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!one_of(op->get_type_info(),
                    ov::op::v0::TensorIterator::get_type_info_static(),
                    ov::op::v5::Loop::get_type_info_static())) {
            errorMessage = "Only opset1 TensorIterator or opset5 Loop operations are supported.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MKLDNNTensorIteratorNode::MKLDNNTensorIteratorNode(const std::shared_ptr<ov::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache) :
        MKLDNNNode(op, eng, cache), m_ngraphOp(op) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }
}

void MKLDNNTensorIteratorNode::getSupportedDescriptors() {
    auto tiOp = ov::as_type_ptr<const ov::op::util::SubGraphOp>(m_ngraphOp);
    if (!tiOp) {
        THROW_ERROR << "cannot be cast to ov::op::util::SubGraphOp";
    }
    const std::shared_ptr<const ov::Function> body = tiOp->get_function();
    m_sub_graph.CreateGraph(body, m_ext_mng, weightCache);

    const auto &inMap = m_sub_graph.GetInputNodesMap();
    for (const auto &param : tiOp->get_function()->get_parameters()) {
        auto inNode = inMap.find(param->get_friendly_name());
        if (inNode != inMap.end()) {
            auto inMem = inNode->second->getChildEdgeAt(0)->getMemoryPtr();
            m_input_mem.push_back(inMem);
        }
    }

    const auto &outMap = m_sub_graph.GetOutputNodesMap();
    for (const auto &out : tiOp->get_function()->get_results()) {
        const auto prev = out->input_value(0);
        const auto inputID = ngraph::op::util::create_ie_output_name(prev);
        auto outNode = outMap.find(inputID);
        if (outNode != outMap.end()) {
            auto outMem = outNode->second->getParentEdgeAt(0)->getMemoryPtr();
            m_output_mem.push_back(outMem);
        }
    }

    // Port map: outputs
    for (const auto& desc : tiOp->get_output_descriptions()) {
        auto body_output_idx = desc->m_body_value_index;

        std::string type_name = desc->get_type_info().name;
        if (type_name == "ConcatOutputDescription") {
            auto output_desc = ov::as_type_ptr<const ov::op::util::SubGraphOp::ConcatOutputDescription>(desc);
            IE_ASSERT(output_desc != nullptr);

            m_outputPortMap.emplace_back(PortMap {
                    static_cast<int>(output_desc->m_output_index), static_cast<int>(body_output_idx),
                    static_cast<int>(output_desc->m_axis), static_cast<int>(output_desc->m_stride),
                    static_cast<int>(output_desc->m_start), static_cast<int>(output_desc->m_end),
                    static_cast<int>(output_desc->m_part_size)});
        } else if (type_name == "BodyOutputDescription") {
            auto output_desc = ov::as_type_ptr<const ov::op::util::SubGraphOp::BodyOutputDescription>(desc);
            IE_ASSERT(output_desc != nullptr);

            m_outputPortMap.emplace_back(PortMap {
                    static_cast<int>(output_desc->m_output_index), static_cast<int>(body_output_idx), -1, 1, 0, -1, 1});
        } else {
            IE_THROW() << "Incorrect type of the output description.";
        }
    }

    // Port map : inputs and back edges
    for (const auto& desc : tiOp->get_input_descriptions()) {
        auto body_input_index = desc->m_body_parameter_index;

        if (auto slice_desc = ov::as_type_ptr<const ov::op::util::SubGraphOp::SliceInputDescription>(desc)) {
            m_inputPortMap.emplace_back(PortMap {
                    static_cast<int>(slice_desc->m_input_index), static_cast<int>(body_input_index),
                    static_cast<int>(slice_desc->m_axis), static_cast<int>(slice_desc->m_stride),
                    static_cast<int>(slice_desc->m_start), static_cast<int>(slice_desc->m_end),
                    static_cast<int>(slice_desc->m_part_size)});
        } else if (auto merge_desc = ov::as_type_ptr<const ov::op::util::SubGraphOp::MergedInputDescription>(desc)) {
            m_inputPortMap.emplace_back(PortMap {
                    static_cast<int>(merge_desc->m_input_index), static_cast<int>(body_input_index), -1, 1, 0, -1, 1});

            auto body_output_idx = merge_desc->m_body_value_index;

            m_backEdges.emplace_back(PortMap {
                    static_cast<int>(body_output_idx), static_cast<int>(body_input_index), -1, 1, 0, -1, 1});
        } else if (auto inv_desc = ov::as_type_ptr<const ov::op::util::SubGraphOp::InvariantInputDescription>(desc)) {
            m_inputPortMap.emplace_back(PortMap {
                    static_cast<int>(inv_desc->m_input_index), static_cast<int>(body_input_index), -1, 1, 0, -1, 1});
        } else {
            THROW_ERROR << "has incorrect type of the input description.";
        }
    }

    if (auto loopOp = ov::as_type_ptr<const ov::op::v5::Loop>(m_ngraphOp)) {
        algorithm = TensorIteratorLoop;
        auto spec_port = loopOp->get_special_body_ports();
        if (spec_port.current_iteration_input_idx != -1) {
            m_loopBodyCurrentIterationIdx.push_back(spec_port.current_iteration_input_idx);
        }
        if (spec_port.body_condition_output_idx != -1) {
            m_loopBodyConditionOutputIdx = spec_port.body_condition_output_idx;
        }
        m_loopTripCountIdx = 0;
        m_loopExecutionConditionIdx = 1;
    } else if (auto ti = ov::as_type_ptr<const ov::op::v0::TensorIterator>(m_ngraphOp)) {
        algorithm = TensorIteratorCommon;
        m_n_iter = ti->get_num_iterations();
    } else {
        THROW_ERROR << "doesn't supported!";
    }
}

void MKLDNNTensorIteratorNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    supportedPrimitiveDescriptors.emplace_back(make_plain_config(m_ngraphOp), impl_desc_type::unknown);
}

void MKLDNNTensorIteratorNode::createPrimitive() {
    if (m_loopBodyConditionOutputIdx == -1)
        m_continue_cond_check.reset(new staticValueCheck(true)); // always true
    if (m_loopExecutionConditionIdx == -1)
        m_initial_cond_check.reset(new staticValueCheck(true));

    if (isDynamic)
        prepareDynamicBuffers();

    if (inputShapesDefined()) {
        if (needPrepareParams())
            prepareParams();
        updateLastInputDims();
    }
}

bool MKLDNNTensorIteratorNode::needPrepareParams() const {
    if (getAlgorithm() == TensorIteratorLoop) {
        const auto tripCountPtr = reinterpret_cast<const uint32_t*>(getParentEdgesAtPort(m_loopTripCountIdx).front()->getMemoryPtr()->GetPtr());
        const auto condPtr = reinterpret_cast<const uint8_t*>(getParentEdgesAtPort(m_loopExecutionConditionIdx).front()->getMemoryPtr()->GetPtr());
        if (tripCountPtr[0] != m_lastUsedTripCount || condPtr[0] != m_lastUsedCond)
            return true;
    }

    return MKLDNNNode::needPrepareParams();
}

void MKLDNNTensorIteratorNode::prepareParams() {
    reshapeSubgraphInput();

    m_first_mappers.clear();
    m_before_mappers.clear();
    m_back_mappers.clear();

    prepareInputPorts();
    prepareInitialCond();
    prepareContinueCond();
    prepareTripCount();
    // special purpose ports
    prepareLoopBodyCurrentIteration();

    if (!isDynamic) {
        prepareOutputPorts();
        prepareBackEdges();
    }
}

void MKLDNNTensorIteratorNode::execute(mkldnn::stream strm) {
    m_sub_graph.ResetInferCount();

    bool continue_cond = m_initial_cond_check->getStatus();
    int max_num_iter = m_trip_count_check->getStatus();

    for (auto &mapper : m_first_mappers)
        mapper->execute(strm);

    // use  "i != max_num_iter" only to allow "-1" works like infinite loop
    for (int i = 0; i != max_num_iter && continue_cond; i++) {
        // copy data to subgraph iteration
        for (auto &mapper : m_before_mappers)
            mapper->execute(strm, i);

        m_sub_graph.Infer();

        continue_cond = m_continue_cond_check->getStatus();

        // copy data from subgraph iteration to outputs
        // or to the next iteration inputs
        for (auto &mapper : m_after_mappers)
            mapper->execute(strm, i);
    }

    for (auto &mapper : m_last_mappers)
        mapper->execute(strm);
}

void MKLDNNTensorIteratorNode::executeDynamicImpl(mkldnn::stream strm) {
    const auto &eng = getEngine();
    m_sub_graph.ResetInferCount();

    bool continue_cond = m_initial_cond_check->getStatus();
    int max_num_iter = m_trip_count_check->getStatus();

    for (auto &mapper : m_first_mappers)
        mapper->execute(strm);

    if (!continue_cond || max_num_iter == 0)
        THROW_ERROR << "has incorrect iteration count for dynamic execution";

    // use  "i != max_num_iter" only to allow "-1" works like infinite loop
    for (int i = 0; i != max_num_iter && continue_cond; i++) {
        // copy data to subgraph iteration
        for (auto &mapper : m_before_mappers)
            mapper->execute(strm, i);
        for (auto &mapper : m_back_mappers)
            mapper->execute(strm, i);

        m_sub_graph.Infer();

        continue_cond = m_continue_cond_check->getStatus();

        for (auto& buffer : m_buffers)
            buffer->execute(strm, i);

        // on the last iteration we shouldn't reshape body inputs and init back edges
        if ((i + 1 != max_num_iter) && continue_cond)
            prepareDynamicBackEdges();
    }

    reshapeAndFillOutput(strm, eng);
}

/* *==============* Prepare reorders, edges between body and TI *==============* */

void MKLDNNTensorIteratorNode::prepareInputPorts() {
    const auto &eng = getEngine();
    for (auto map_rule : m_inputPortMap) {
        auto &from_mem = getParentEdgesAtPort(map_rule.m_from)[0]->getMemoryPtr();
        auto &to_mem = m_input_mem[map_rule.m_to];

        if (map_rule.m_axis == -1)
            m_first_mappers.emplace_back(std::make_shared<BackEdgePortHelper>(from_mem, to_mem, eng));
        else
            m_before_mappers.emplace_back(std::make_shared<PortIteratorHelper>(from_mem, to_mem, true, map_rule, eng));
    }
}

void MKLDNNTensorIteratorNode::prepareOutputPorts() {
    const auto &eng = getEngine();
    for (auto map_rule : m_outputPortMap) {
        auto &to_mem = getChildEdgesAtPort(map_rule.m_from)[0]->getMemoryPtr();
        auto &from_mem = m_output_mem[map_rule.m_to];

        if (map_rule.m_axis == -1)
            m_last_mappers.emplace_back(std::make_shared<BackEdgePortHelper>(from_mem, to_mem, eng));
        else
            m_after_mappers.emplace_back(std::make_shared<PortIteratorHelper>(from_mem, to_mem, false, map_rule, eng));
    }
}

void MKLDNNTensorIteratorNode::prepareBackEdges() {
    const auto &eng = getEngine();
    for (auto map_rule : m_backEdges) {
        auto from_mem = m_output_mem[map_rule.m_from];
        auto to_mem = m_input_mem[map_rule.m_to];

        m_before_mappers.emplace_back(std::make_shared<BackEdgePortHelper>(from_mem, to_mem, eng));
    }
}

void MKLDNNTensorIteratorNode::prepareDynamicBackEdges() {
    const auto &eng = getEngine();
    m_back_mappers.clear();
    for (auto map_rule : m_backEdges) {
        auto from_mem = m_output_mem[map_rule.m_from];
        auto to_mem = m_input_mem[map_rule.m_to];

        // need to reshape body inputs by output if body has internal dynamism
        const auto &currDesc = to_mem->getDesc();
        if (currDesc.getShape().isDynamic() || currDesc.getShape().getStaticDims() != from_mem->getStaticDims()) {
            const auto memDesc = getBaseMemDescAtInputPort(map_rule.m_from)->cloneWithNewDims(from_mem->getStaticDims());
            to_mem->redefineDesc(*memDesc);
        }

        m_back_mappers.emplace_back(std::make_shared<BackEdgePortHelper>(from_mem, to_mem, eng));
    }
}

void MKLDNNTensorIteratorNode::prepareDynamicBuffers() {
    for (auto map_rule : m_outputPortMap) {
        if (map_rule.m_axis != -1) {
            auto &to_mem = getChildEdgesAtPort(map_rule.m_from)[0]->getMemoryPtr();
            auto &from_mem = m_output_mem[map_rule.m_to];
            m_buffers.emplace_back(std::make_shared<DynamicBuffer>(from_mem, to_mem, map_rule));
        }
    }
}

void MKLDNNTensorIteratorNode::prepareLoopBodyCurrentIteration() {
    const auto &eng = getEngine();
    for (auto idx : m_loopBodyCurrentIterationIdx) {
        auto to_mem = m_input_mem[idx];
        m_before_mappers.emplace_back(std::make_shared<IterCountPortHelper>(to_mem, eng));
    }
}

void MKLDNNTensorIteratorNode::prepareContinueCond() {
    if (m_loopBodyConditionOutputIdx != -1 || !m_continue_cond_check) {
        auto mem = m_output_mem[m_loopBodyConditionOutputIdx];
        m_continue_cond_check.reset(new asBoolCheck(mem));
    }
}

void MKLDNNTensorIteratorNode::prepareInitialCond() {
    if (m_loopExecutionConditionIdx != -1 || !m_initial_cond_check) {
        auto mem = getParentEdgesAtPort(m_loopExecutionConditionIdx)[0]->getMemoryPtr();
        m_initial_cond_check.reset(new asBoolCheck(mem));
        m_lastUsedCond = m_initial_cond_check->getStatus();
    }
}

void MKLDNNTensorIteratorNode::prepareTripCount() {
    if (m_loopTripCountIdx == -1) {
        m_trip_count_check.reset(new staticValueCheck(m_n_iter)); // use statically calculated num of iteration
    } else {
        auto mem = getParentEdgesAtPort(m_loopTripCountIdx)[0]->getMemoryPtr();
        m_trip_count_check.reset(new asIntCheck(mem));
    }
    m_lastUsedTripCount = m_trip_count_check->getStatus();
}

/* *==============* *==============* *==============* *==============* *==============* */

void MKLDNNTensorIteratorNode::reshapeSubgraphInput() {
    for (auto map_rule : m_inputPortMap) {
        auto &from_mem = getParentEdgesAtPort(map_rule.m_from)[0]->getMemoryPtr();
        auto &to_mem = m_input_mem[map_rule.m_to];

        auto new_dims = from_mem->getStaticDims();
        if (map_rule.m_axis != -1)
            new_dims[map_rule.m_axis] = abs(map_rule.m_stride);

        const auto &currDesc = to_mem->getDesc();
        if (currDesc.getShape().isStatic() && currDesc.getShape().getStaticDims() == new_dims)
            continue;

        const auto memDesc = std::make_shared<CpuBlockedMemoryDesc>(currDesc.getPrecision(), Shape(new_dims));
        to_mem->redefineDesc(*memDesc);
    }
}

void MKLDNNTensorIteratorNode::reshapeAndFillOutput(mkldnn::stream strm, const mkldnn::engine& eng) {
    for (auto map_rule : m_outputPortMap) {
        if (map_rule.m_axis == -1) {
            auto &to_mem = getChildEdgesAtPort(map_rule.m_from)[0]->getMemoryPtr();
            auto &from_mem = m_output_mem[map_rule.m_to];

            const auto &currDesc = to_mem->getDesc();
            if (!currDesc.getShape().isStatic() || currDesc.getShape().getStaticDims() != from_mem->getStaticDims()) {
                const auto memDesc = getBaseMemDescAtOutputPort(map_rule.m_from)->cloneWithNewDims(
                        from_mem->getStaticDims());
                to_mem->redefineDesc(*memDesc);
            }

            PortMapHelper *mapper = new BackEdgePortHelper(from_mem, to_mem, eng);
            mapper->execute(strm);
        }
    }

    for (auto buffer : m_buffers) {
        buffer->transfer(strm, this);
    }
}

bool MKLDNNTensorIteratorNode::created() const {
    return getType() == TensorIterator;
}
REG_MKLDNN_PRIM_FOR(MKLDNNTensorIteratorNode, TensorIterator);
