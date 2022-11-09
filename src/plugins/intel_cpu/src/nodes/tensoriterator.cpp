// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tensoriterator.h"

#include <string>
#include <vector>
#include <dnnl_extension_utils.h>
#include <ie_ngraph_utils.hpp>
#include <utils/general_utils.h>
#include "common/blocked_desc_creator.h"
#include "utils/ngraph_utils.hpp"
#include "transformations/utils/utils.hpp"
#include "common/cpu_memcpy.h"

using namespace dnnl;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

namespace ov {
namespace intel_cpu {
namespace node {

#define THROW_ERROR IE_THROW() << getTypeStr() << " layer with name '" << getName() << "' "

static NodeConfig make_plain_config(const std::shared_ptr<ov::Node>& op) {
    NodeConfig config;

    for (size_t i = 0; i < op->get_input_size(); i++) {
        const auto &origShape = op->get_input_partial_shape(i);
        const auto& shape = Shape(origShape.rank().get_length() == 0 ? ov::PartialShape{1} : origShape);
        const auto prec = InferenceEngine::details::convertPrecision(op->get_input_element_type(i));

        PortConfig data_conf {};
        auto descCreator = BlockedDescCreator::getCommonCreators().at(LayoutType::ncsp);
        data_conf.setMemDesc(descCreator->createSharedDesc(prec, shape));
        config.inConfs.push_back(data_conf);
    }

    for (size_t i = 0; i < op->get_output_size(); i++) {
        const auto &origShape = op->get_output_partial_shape(i);
        const auto& shape = Shape(origShape.rank().get_length() == 0 ? ov::PartialShape{1} : origShape);
        const auto prec = InferenceEngine::details::convertPrecision(op->get_output_element_type(i));

        PortConfig data_conf {};
        auto descCreator = BlockedDescCreator::getCommonCreators().at(LayoutType::ncsp);
        data_conf.setMemDesc(descCreator->createSharedDesc(prec, shape));
        config.outConfs.push_back(data_conf);
    }

    config.dynBatchSupport = true;
    return config;
}

static void redefineToMemories(const std::vector<MemoryPtr>& to_mems, MemoryDescPtr new_desc) {
    const auto &currDesc = to_mems.front()->getDesc();
    if (currDesc.getShape().isDynamic() || currDesc.getShape().getStaticDims() != new_desc->getShape().getStaticDims()) {
        // TODO : check the entire dstMemPtrs usage considering the proper memory sharing
        for (size_t j = 0; j < to_mems.size(); j++) {
            to_mems[j]->redefineDesc(new_desc);
        }
    }
}

// this method get all memory ptrs of childs of one port to redefine descs for them
static std::vector<MemoryPtr> getToMemories(const Node* node, const size_t port) {
    std::vector<MemoryPtr> memories;
    for (auto& edge : node->getChildEdgesAtPort(port))
        memories.push_back(edge->getMemoryPtr());
    return memories;
}

static void nullifyUndefinedDims(VectorDims& dims) {
    std::transform(dims.begin(), dims.end(), dims.begin(), [](const size_t& dim) {
        return dim == Shape::UNDEFINED_DIM ? 0 : dim;
    });
}

class PortIteratorHelper : public PortMapHelper {
public:
    PortIteratorHelper(const MemoryPtr &from, const MemoryPtr &to, bool sliced_src,
                       const PortMap &slice_rule, const dnnl::engine& eng)
                       : sliced_src(sliced_src) {
        const auto &full_blob = sliced_src ? from : to;
        const auto &part_blob = !sliced_src ? from : to;

        auto axis = slice_rule.axis;
        auto stride = slice_rule.stride;

        auto full_dims = full_blob->GetShape().getStaticDims();
        auto part_dims = part_blob->GetShape().getStaticDims();

        auto abs_stride = std::abs(stride);
        auto sign_of_stride = stride < 0.0f ? -1 : 1;

        iter_count = full_dims[axis] / abs_stride;

        full_dims[axis] = abs_stride;
        IE_ASSERT(full_dims == part_dims) << "Shape mismatch for tensor iterator port";

        // make chunk view
        auto chunk_desc = full_blob->GetDescWithType<DnnlMemoryDesc>()->getDnnlDesc();
        chunk_desc.data.dims[axis] = abs_stride;
        chunk_desc.data.padded_dims[axis] = abs_stride;  // TODO: asamption that plain tensor

        full_mem = full_blob->GetPrimitive();
        const auto full_mem_handler = full_mem.get_data_handle();
        dnnl::memory chunk_mem = {chunk_desc, eng, full_mem_handler};

        auto elem_size = DnnlExtensionUtils::sizeOfDataType(dnnl::memory::data_type(chunk_desc.data.data_type));

        chunk_stride_in_byte = chunk_desc.data.format_desc.blocking.strides[axis] * elem_size * abs_stride;
        chunk_offset_in_byte = sign_of_stride < 0 ? (iter_count - 1) * chunk_stride_in_byte : 0;
        chunk_stride_in_byte *= sign_of_stride;

        if (sliced_src) {
            mem_holder_src = chunk_mem;
            mem_holder_dst = to->GetPrimitive();
        } else {
            mem_holder_src = from->GetPrimitive();
            mem_holder_dst = chunk_mem;
        }
        reorder = {mem_holder_src, mem_holder_dst};
    }

    void execute(dnnl::stream strm, int iter) override {
        IE_ASSERT(iter >= 0 && iter < iter_count);

        auto &chunk_mem = sliced_src ? mem_holder_src : mem_holder_dst;
        chunk_mem.set_data_handle(static_cast<uint8_t *>(full_mem.get_data_handle()) +
                                          chunk_offset_in_byte + chunk_stride_in_byte * iter);

        reorder.execute(strm, mem_holder_src, mem_holder_dst);
    }

private:
    ptrdiff_t chunk_stride_in_byte = 0;
    ptrdiff_t chunk_offset_in_byte = 0;

    bool sliced_src;
    dnnl::memory full_mem;

    int iter_count;
};

class BackEdgePortHelper : public PortMapHelper {
public:
    BackEdgePortHelper(const MemoryPtr &from, const MemoryPtr &to, const dnnl::engine& eng) {
        mem_holder_src = from->GetPrimitive();
        mem_holder_dst = to->GetPrimitive();
        reorder = {mem_holder_src, mem_holder_dst};
    }

    void execute(dnnl::stream strm, int iter = -1) override {
        if (iter != 0) {
            reorder.execute(strm, mem_holder_src, mem_holder_dst);
        }
    }
};

class IterCountPortHelper : public PortMapHelper {
public:
    IterCountPortHelper(const MemoryPtr &to, const dnnl::engine& eng) {
        // Only scalar I32 tensor is supported
        IE_ASSERT(to->GetDataType() == memory::data_type::s32);
        IE_ASSERT(to->GetShape() == Shape(VectorDims{1}));
        mem_holder_dst = to->GetPrimitive();
    }

    void execute(dnnl::stream strm, int n_iter) override {
        auto mem = mem_holder_dst;
        auto data_ptr = static_cast<uint32_t*>(mem.get_data_handle());
        if (data_ptr == nullptr) {
            IE_THROW() << "TensorIterator node has not allocated memory for IterCountPortHelper";
        }
        *data_ptr = n_iter;
    }
};

class asBoolCheck : public PortChecker {
public:
    asBoolCheck(const MemoryPtr &mem) {
        IE_ASSERT(mem->GetDataType() == memory::data_type::u8);
        IE_ASSERT(mem->GetShape() == Shape(InferenceEngine::SizeVector{1}));
        mem_holder = mem->GetPrimitive();
    }

    int getStatus() override {
        auto data_ptr = static_cast<uint8_t*>(mem_holder.get_data_handle());
        if (data_ptr == nullptr) {
            IE_THROW() << "TensorIterator node has not allocated memory for asBoolCheck";
        }
        return *data_ptr == static_cast<uint8_t>(0) ? 0 : 1;
    }
};

class asIntCheck : public PortChecker {
public:
    asIntCheck(const MemoryPtr &mem) {
        IE_ASSERT(mem->GetDataType() == memory::data_type::s32);
        IE_ASSERT(mem->GetShape() == Shape(InferenceEngine::SizeVector{1}));
        mem_holder = mem->GetPrimitive();
    }

    int getStatus() override {
        auto data_ptr = static_cast<uint32_t*>(mem_holder.get_data_handle());
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

DynamicBuffer::DynamicBuffer(const MemoryPtr &from_, const std::vector<MemoryPtr> &to_,
                             const PortMap &map_rule_) : from(from_), to(to_), map_rule(map_rule_) {
    elem_size = DnnlExtensionUtils::sizeOfDataType(from->GetDataType());
}

void DynamicBuffer::execute(const dnnl::engine& eng, const int iter) {
    if (iter == 0) {
        init(eng);
        return;
    }

    auto new_buffer = create_buffer(eng);
    move_buffer(new_buffer);
    move_data();
}

void DynamicBuffer::init(const dnnl::engine& eng) {
    chunk_offset_in_byte = 0;
    buffer_offset_in_byte = 0;

    const auto axis = map_rule.axis;
    const auto stride = map_rule.stride;
    const auto abs_stride = std::abs(stride);

    auto src_mem = from->GetPrimitive();
    auto src_desc = src_mem.get_desc();
    auto dims = src_desc.dims();

    if (dims[axis] != abs_stride)
        IE_THROW() << "TensorIterator (Loop) has incorrect output shape[axis] after iteration for concatenation. " << abs_stride <<
                   " is expected, but actual: " << dims[axis];

    count = std::accumulate(dims.begin(), dims.begin() + map_rule.axis, size_t(1), std::multiplies<size_t>());
    len = std::accumulate(dims.begin() + map_rule.axis + 1, dims.end(), elem_size, std::multiplies<size_t>());
    mem_holder_buffer.reset(new memory(src_desc, eng));
    copy(reinterpret_cast<const uint8_t*>(from->GetPtr()), get_ptr(*mem_holder_buffer.get()), 0, 0, 1, from->GetSize());
}

std::shared_ptr<dnnl::memory> DynamicBuffer::create_buffer(const dnnl::engine& eng) {
    const auto axis = map_rule.axis;
    const auto stride = map_rule.stride;
    const auto abs_stride = std::abs(stride);

    const auto old_desc = mem_holder_buffer->get_desc();
    auto dims = old_desc.dims();

    if (from->getStaticDims()[axis] != abs_stride)
        IE_THROW() << "TensorIterator (Loop) has incorrect output shape[axis] after iteration for concatenation. " << abs_stride <<
        " is expected, but actual: " << from->getStaticDims()[axis];

    dims[axis] += abs_stride;
    dnnl::memory::desc new_buffer_desc(dims, old_desc.data_type(), DnnlExtensionUtils::GetPlainFormatByRank(dims.size()));

    if (stride > 0.0f) {
        chunk_offset_in_byte += new_buffer_desc.data.format_desc.blocking.strides[axis] * elem_size * abs_stride;
    } else {
        buffer_offset_in_byte = from->GetPrimitive().get_desc().data.format_desc.blocking.strides[axis] * elem_size * abs_stride;
    }

    return std::make_shared<dnnl::memory>(new_buffer_desc, eng);
}

void DynamicBuffer::move_buffer(std::shared_ptr<dnnl::memory> new_buffer) {
    const auto axis = map_rule.axis;
    const auto src_stride = mem_holder_buffer->get_desc().dims()[axis] * len;
    const auto dst_stride = new_buffer->get_desc().dims()[axis] * len;

    copy(get_ptr(*mem_holder_buffer.get()), get_ptr(*new_buffer.get()) + buffer_offset_in_byte,
         src_stride, dst_stride, count, src_stride);
    mem_holder_buffer = new_buffer;
}

void DynamicBuffer::move_data() {
    const auto axis = map_rule.axis;
    const auto src_stride = abs(map_rule.stride) * len;
    const auto dst_stride = mem_holder_buffer->get_desc().dims()[axis] * len;

    copy(reinterpret_cast<const uint8_t*>(from->GetPtr()), get_ptr(*mem_holder_buffer.get()) + chunk_offset_in_byte,
         src_stride, dst_stride, count, src_stride);
}

void DynamicBuffer::transfer(const Node* node) {
    if (mem_holder_buffer) {
        const auto desc = node->getBaseMemDescAtOutputPort(map_rule.from)->cloneWithNewDims(
                DnnlExtensionUtils::convertToVectorDims(mem_holder_buffer->get_desc().dims()));
        redefineToMemories(to, desc);

        copy(get_ptr(*mem_holder_buffer.get()), reinterpret_cast<uint8_t*>(to.front()->GetPtr()), 0, 0, 1, to.front()->GetSize());
    } else {
        VectorDims newDims = to.front()->GetShape().getDims();
        nullifyUndefinedDims(newDims);

        const auto desc = node->getBaseMemDescAtOutputPort(map_rule.from)->cloneWithNewDims(newDims);
        redefineToMemories(to, desc);
    }

    mem_holder_buffer.reset();
}

void DynamicBuffer::copy(const uint8_t* src, uint8_t* dst, const size_t src_stride, const size_t dst_stride, const size_t count, const size_t len) {
    parallel_for(count, [&](const size_t i) {
        cpu_memcpy(&dst[i * dst_stride], &src[i * src_stride], len);
    });
}

uint8_t* DynamicBuffer::get_ptr(dnnl::memory& prim) {
    auto ptr = static_cast<uint8_t*>(prim.get_data_handle());
    auto md = prim.get_desc().data;
    dnnl::impl::memory_desc_wrapper wrapper(md);
    return ptr + wrapper.offset0() * wrapper.data_type_size();
}

bool TensorIterator::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
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

TensorIterator::TensorIterator(const std::shared_ptr<ov::Node>& op, const dnnl::engine& eng, WeightsSharing::Ptr &cache) :
        Node(op, eng, cache), ngraphOp(op) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }
}

void TensorIterator::getSupportedDescriptors() {
    auto tiOp = ov::as_type_ptr<const ov::op::util::SubGraphOp>(ngraphOp);
    if (!tiOp) {
        THROW_ERROR << "cannot be cast to ov::op::util::SubGraphOp";
    }
    const std::shared_ptr<const ov::Model> body = tiOp->get_function();
    sub_graph.CreateGraph(body, ext_mng, weightCache, sharedMutex);

    const auto &inMap = sub_graph.GetInputNodesMap();
    for (const auto &param : tiOp->get_function()->get_parameters()) {
        auto inNode = inMap.find(param->get_friendly_name());
        if (inNode != inMap.end()) {
            input_mems.push_back(getToMemories(inNode->second.get(), 0));
        }
    }

    const auto &outMap = sub_graph.GetOutputNodesMap();
    for (const auto &out : tiOp->get_function()->get_results()) {
        const auto prev = out->input_value(0);
        const auto inputID = ngraph::op::util::create_ie_output_name(prev);
        auto outNode = outMap.find(inputID);
        if (outNode != outMap.end()) {
            auto outMem = outNode->second->getParentEdgeAt(0)->getMemoryPtr();
            output_mem.push_back(outMem);
        }
    }

    // Port map: outputs
    for (const auto& desc : tiOp->get_output_descriptions()) {
        auto body_output_idx = desc->m_body_value_index;

        std::string type_name = desc->get_type_info().name;
        if (type_name == "ConcatOutputDescription") {
            auto output_desc = ov::as_type_ptr<const ov::op::util::SubGraphOp::ConcatOutputDescription>(desc);
            IE_ASSERT(output_desc != nullptr);

            outputPortMap.emplace_back(PortMap {
                    static_cast<int>(output_desc->m_output_index), static_cast<int>(body_output_idx),
                    static_cast<int>(output_desc->m_axis), static_cast<int>(output_desc->m_stride),
                    static_cast<int>(output_desc->m_start), static_cast<int>(output_desc->m_end),
                    static_cast<int>(output_desc->m_part_size)});
        } else if (type_name == "BodyOutputDescription") {
            auto output_desc = ov::as_type_ptr<const ov::op::util::SubGraphOp::BodyOutputDescription>(desc);
            IE_ASSERT(output_desc != nullptr);

            outputPortMap.emplace_back(PortMap {
                    static_cast<int>(output_desc->m_output_index), static_cast<int>(body_output_idx), -1, 1, 0, -1, 1});
        } else {
            IE_THROW() << "Incorrect type of the output description.";
        }
    }

    // Port map : inputs and back edges
    for (const auto& desc : tiOp->get_input_descriptions()) {
        auto body_input_index = desc->m_body_parameter_index;

        if (auto slice_desc = ov::as_type_ptr<const ov::op::util::SubGraphOp::SliceInputDescription>(desc)) {
            inputPortMap.emplace_back(PortMap {
                    static_cast<int>(slice_desc->m_input_index), static_cast<int>(body_input_index),
                    static_cast<int>(slice_desc->m_axis), static_cast<int>(slice_desc->m_stride),
                    static_cast<int>(slice_desc->m_start), static_cast<int>(slice_desc->m_end),
                    static_cast<int>(slice_desc->m_part_size)});
        } else if (auto merge_desc = ov::as_type_ptr<const ov::op::util::SubGraphOp::MergedInputDescription>(desc)) {
            inputPortMap.emplace_back(PortMap {
                    static_cast<int>(merge_desc->m_input_index), static_cast<int>(body_input_index), -1, 1, 0, -1, 1});

            auto body_output_idx = merge_desc->m_body_value_index;

            backEdges.emplace_back(PortMap {
                    static_cast<int>(body_output_idx), static_cast<int>(body_input_index), -1, 1, 0, -1, 1});
        } else if (auto inv_desc = ov::as_type_ptr<const ov::op::util::SubGraphOp::InvariantInputDescription>(desc)) {
            inputPortMap.emplace_back(PortMap {
                    static_cast<int>(inv_desc->m_input_index), static_cast<int>(body_input_index), -1, 1, 0, -1, 1});
        } else {
            THROW_ERROR << "has incorrect type of the input description.";
        }
    }

    if (auto loopOp = ov::as_type_ptr<const ov::op::v5::Loop>(ngraphOp)) {
        algorithm = Algorithm::TensorIteratorLoop;
        auto spec_port = loopOp->get_special_body_ports();
        if (spec_port.current_iteration_input_idx != -1) {
            loopBodyCurrentIterationIdx.push_back(spec_port.current_iteration_input_idx);
        }
        if (spec_port.body_condition_output_idx != -1) {
            loopBodyConditionOutputIdx = spec_port.body_condition_output_idx;
        }
        loopTripCountIdx = 0;
        loopExecutionConditionIdx = 1;
    } else if (auto ti = ov::as_type_ptr<const ov::op::v0::TensorIterator>(ngraphOp)) {
        algorithm = Algorithm::TensorIteratorCommon;
    } else {
        THROW_ERROR << "isn't supported!";
    }
}

void TensorIterator::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    supportedPrimitiveDescriptors.emplace_back(make_plain_config(ngraphOp), impl_desc_type::unknown);
}

void TensorIterator::createPrimitive() {
    if (loopBodyConditionOutputIdx == -1)
        continue_cond_check.reset(new staticValueCheck(true)); // always true
    if (loopExecutionConditionIdx == -1) {
        initial_cond_check.reset(new staticValueCheck(true));
        lastUsedCond = initial_cond_check->getStatus();
    }

    if (isDynamicNode())
        prepareDynamicBuffers();

    Node::createPrimitive();
}

bool TensorIterator::needPrepareParams() const {
    if (getAlgorithm() == Algorithm::TensorIteratorLoop) {
        const auto tripCountPtr = reinterpret_cast<const uint32_t*>(getParentEdgesAtPort(loopTripCountIdx).front()->getMemoryPtr()->GetPtr());
        const auto condPtr = reinterpret_cast<const uint8_t*>(getParentEdgesAtPort(loopExecutionConditionIdx).front()->getMemoryPtr()->GetPtr());
        if (tripCountPtr[0] != lastUsedTripCount || static_cast<bool>(condPtr[0]) != lastUsedCond)
            return true;
    }

    return Node::needPrepareParams();
}

void TensorIterator::prepareParams() {
    prepareTripCount();
    prepareInitialCond();

    first_mappers.clear();
    before_mappers.clear();
    back_mappers.clear();

    if ((lastUsedCond && lastUsedTripCount != 0) || !isDynamicNode()) {
        reshapeSubgraphInput();

        prepareInputPorts();
        prepareContinueCond();
        prepareLoopBodyCurrentIteration();

        if (!isDynamicNode()) {
            prepareOutputPorts();
            prepareBackEdges();
        }
    }
}

void TensorIterator::execute(dnnl::stream strm) {
    sub_graph.ResetInferCount();

    bool continue_cond = initial_cond_check->getStatus();
    int max_num_iter = trip_count_check->getStatus();

    for (auto &mapper : first_mappers)
        mapper->execute(strm);

    // use  "i != max_num_iter" only to allow "-1" works like infinite loop
    for (int i = 0; i != max_num_iter && continue_cond; i++) {
        // copy data to subgraph iteration
        for (auto &mapper : before_mappers)
            mapper->execute(strm, i);

        sub_graph.Infer();

        continue_cond = continue_cond_check->getStatus();

        // copy data from subgraph iteration to outputs
        // or to the next iteration inputs
        for (auto &mapper : after_mappers)
            mapper->execute(strm, i);
    }

    for (auto &mapper : last_mappers)
        mapper->execute(strm);
}

void TensorIterator::executeDynamicImpl(dnnl::stream strm) {
    const auto &eng = getEngine();
    sub_graph.ResetInferCount();

    bool continue_cond = initial_cond_check->getStatus();
    int max_num_iter = trip_count_check->getStatus();

    for (auto &mapper : first_mappers)
        mapper->execute(strm);

    // use  "i != max_num_iter" only to allow "-1" works like infinite loop
    for (int i = 0; i != max_num_iter && continue_cond; i++) {
        // copy data to subgraph iteration
        for (auto &mapper : before_mappers)
            mapper->execute(strm, i);
        for (auto &mapper : back_mappers)
            mapper->execute(strm, i);

        sub_graph.Infer();

        continue_cond = continue_cond_check->getStatus();

        for (auto& buffer : buffers)
            buffer->execute(eng, i);

        // on the last iteration we shouldn't reshape body inputs and init back edges
        if ((i + 1 != max_num_iter) && continue_cond)
            prepareDynamicBackEdges();
    }

    reshapeAndFillOutput(strm);
}

/* *==============* Prepare reorders, edges between body and TI *==============* */

void TensorIterator::prepareInputPorts() {
    const auto &eng = getEngine();
    for (auto map_rule : inputPortMap) {
        auto &from_mem = getParentEdgesAtPort(map_rule.from)[0]->getMemoryPtr();
        auto &to_mem = input_mems[map_rule.to].front();  // first memory is enough to access the shared underlying physical memory

        if (map_rule.axis == -1)
            first_mappers.emplace_back(std::make_shared<BackEdgePortHelper>(from_mem, to_mem, eng));
        else
            before_mappers.emplace_back(
                    std::make_shared<PortIteratorHelper>(from_mem, to_mem, true, map_rule, eng));
    }
}

void TensorIterator::prepareOutputPorts() {
    const auto &eng = getEngine();
    for (auto map_rule : outputPortMap) {
        auto &to_mem = getChildEdgesAtPort(map_rule.from)[0]->getMemoryPtr();
        auto &from_mem = output_mem[map_rule.to];

        if (map_rule.axis == -1)
            last_mappers.emplace_back(std::make_shared<BackEdgePortHelper>(from_mem, to_mem, eng));
        else
            after_mappers.emplace_back(std::make_shared<PortIteratorHelper>(from_mem, to_mem, false, map_rule, eng));
    }
}

void TensorIterator::prepareBackEdges() {
    const auto &eng = getEngine();
    for (auto map_rule : backEdges) {
        auto from_mem = output_mem[map_rule.from];
        auto to_mem = input_mems[map_rule.to].front();

        before_mappers.emplace_back(std::make_shared<BackEdgePortHelper>(from_mem, to_mem, eng));
    }
}

void TensorIterator::prepareDynamicBackEdges() {
    const auto &eng = getEngine();
    back_mappers.clear();
    for (auto map_rule : backEdges) {
        auto from_mem = output_mem[map_rule.from];
        auto to_mems = input_mems[map_rule.to];

        redefineToMemories(to_mems, from_mem->getDescPtr());

        // first memory is enough to get common memory ptr
        back_mappers.emplace_back(std::make_shared<BackEdgePortHelper>(from_mem, to_mems.front(), eng));
    }
}

void TensorIterator::prepareDynamicBuffers() {
    for (auto map_rule : outputPortMap) {
        if (map_rule.axis != -1) {
            auto to_mems = getToMemories(this, map_rule.from);
            auto &from_mem = output_mem[map_rule.to];
            buffers.emplace_back(std::make_shared<DynamicBuffer>(from_mem, to_mems, map_rule));
        }
    }
}

void TensorIterator::prepareLoopBodyCurrentIteration() {
    const auto &eng = getEngine();
    for (auto idx : loopBodyCurrentIterationIdx) {
        auto to_mem = input_mems[idx].front();  // first memory is enough to get common memory ptr
        before_mappers.emplace_back(std::make_shared<IterCountPortHelper>(to_mem, eng));
    }
}

void TensorIterator::prepareContinueCond() {
    if (loopBodyConditionOutputIdx != -1 || !continue_cond_check) {
        auto mem = output_mem[loopBodyConditionOutputIdx];
        continue_cond_check.reset(new asBoolCheck(mem));
    }
}

void TensorIterator::prepareInitialCond() {
    if (loopExecutionConditionIdx != -1 || !initial_cond_check) {
        auto mem = getParentEdgesAtPort(loopExecutionConditionIdx)[0]->getMemoryPtr();
        initial_cond_check.reset(new asBoolCheck(mem));
        lastUsedCond = initial_cond_check->getStatus();
    }
}

void TensorIterator::prepareTripCount() {
    if (loopTripCountIdx == -1) {
        trip_count_check.reset(new staticValueCheck(getNumIteration(inputPortMap, outputPortMap)));
    } else {
        auto mem = getParentEdgesAtPort(loopTripCountIdx)[0]->getMemoryPtr();
        trip_count_check.reset(new asIntCheck(mem));
    }
    lastUsedTripCount = trip_count_check->getStatus();
}

/* *==============* *==============* *==============* *==============* *==============* */

void TensorIterator::reshapeSubgraphInput() {
    for (auto map_rule : inputPortMap) {
        auto &from_mem = getParentEdgesAtPort(map_rule.from)[0]->getMemoryPtr();
        auto &to_mems = input_mems[map_rule.to];
        auto new_dims = from_mem->getStaticDims();
        if (map_rule.axis != -1)
            new_dims[map_rule.axis] = abs(map_rule.stride);

        const auto desc = std::make_shared<CpuBlockedMemoryDesc>(to_mems.front()->getDesc().getPrecision(), Shape(new_dims));
        redefineToMemories(to_mems, desc);
    }
}

void TensorIterator::reshapeAndFillOutput(dnnl::stream strm) {
    auto eng = strm.get_engine();
    for (auto map_rule : outputPortMap) {
        if (map_rule.axis == -1) {
            auto to_mems = getToMemories(this, map_rule.from);
            auto &from_mem = output_mem[map_rule.to];

            // if Loop or TI isn't executed we should fill dynamic dims by zero
            auto newShape = from_mem->GetShape();
            auto newDims = newShape.getDims();
            nullifyUndefinedDims(newDims);

            const bool hasZeroDims = std::count(std::begin(newDims), std::end(newDims), 0) > 0;
            const auto desc = getBaseMemDescAtOutputPort(map_rule.from)->cloneWithNewDims(newDims, hasZeroDims);
            redefineToMemories(to_mems, desc);

            if (!newShape.isDynamic()) {
                BackEdgePortHelper mapper(from_mem, to_mems.front(), eng);
                mapper.execute(strm);
            }
        }
    }

    for (auto buffer : buffers) {
        buffer->transfer(this);
    }
}

int TensorIterator::getNumIteration(const std::vector<PortMap>& inputPortMap, const std::vector<PortMap>& outputPortMap) const {
    const auto isIterable = [](const PortMap& rule) {
        return rule.axis != -1;
    };

    const auto getNumIterations = [this](const PortMap& rule, const std::vector<size_t>& dimensions) -> int {
        const auto axis = rule.axis;
        if (axis < 0 || static_cast<std::size_t>(axis) >= dimensions.size()) {
            THROW_ERROR << ": Invalid \"axis\" value in an iteration component: "
                        << rule.axis  << ", dimensions number = " << dimensions.size() << " (out of range)";
        }
        const auto space = dimensions[axis];
        const int start = static_cast<int>((rule.start < 0 ? (space + 1) : 0) + rule.start);
        const int end   = static_cast<int>((rule.end   < 0 ? (space + 1) : 0) + rule.end);

        const auto stride = rule.stride;
        if (stride == 0) {
            THROW_ERROR << ": Invalid \"stride\" value in an iteration component: " << rule.stride << " (infinite loop)";
        }
        const auto step = std::abs(stride);

        const auto src = stride < 0 ? end : start;
        const auto dst = stride < 0 ? start : end;
        const auto length = dst - src;
        if (src < 0 || src >= dst || dst > static_cast<int64_t>(space) || length < step) {
            THROW_ERROR << ": Invalid \"start\",\"stride\",\"end\" values in an iteration component"
                        << ": \"start\" = " << rule.start << ", \"stride\" = " << rule.stride  << ", \"end\" = " << rule.end;
        }

        if (length % step != 0) {
            THROW_ERROR << ": Each iteration must be the same size: length (" << length << ") is not divisible by step (" << step << ")";
        }

        return static_cast<int>(length / step);
    };


    int numIterations = 1;
    bool isDefault = true;
    for (const auto& rule : inputPortMap) {
        const auto& dims = getParentEdgesAtPort(rule.from)[0]->getMemoryPtr()->getStaticDims();
        if (!isIterable(rule)) {
            continue;
        }

        if (rule.from < 0 || rule.from >= static_cast<int64_t>(inputShapes.size())) {
            THROW_ERROR << ": Invalid \"from\" value: \"from\" = " << rule.from
                        << " inputs number = " << inputShapes.size() << " (out of range)";
        }

        const auto currentNumIterations = getNumIterations(rule, dims);
        if (isDefault) {
            isDefault = false;
            numIterations = currentNumIterations;
        } else if (numIterations != currentNumIterations) {
            THROW_ERROR << ": There are at least two different iterations numbers: " << numIterations << " and " << currentNumIterations;
        }
    }

    for (const auto& rule : outputPortMap) {
        const auto& dims = getBaseMemDescAtOutputPort(rule.from)->getShape().getDims();
        if (!isIterable(rule)) {
            continue;
        }

        if (dims[rule.axis] == Shape::UNDEFINED_DIM)
            continue;

        if (rule.from < 0 || rule.from >= static_cast<int64_t>(outputShapes.size())) {
            THROW_ERROR << ": Invalid \"from\" value: \"from\" = " << rule.from
                        << " inputs number = " << outputShapes.size() << " (out of range)";
        }

        const auto currentNumIterations = getNumIterations(rule, dims);
        if (isDefault) {
            isDefault = false;
            numIterations = currentNumIterations;
        } else if (numIterations != currentNumIterations) {
            THROW_ERROR << ": There are at least two different iterations numbers: " << numIterations << " and " << currentNumIterations;
        }
    }

    return numIterations;
}

bool TensorIterator::created() const {
    return getType() == Type::TensorIterator;
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
