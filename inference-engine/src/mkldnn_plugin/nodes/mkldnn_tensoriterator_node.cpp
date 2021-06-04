// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_tensoriterator_node.h"

#include <string>
#include <vector>
#include <map>
#include <mkldnn_extension_utils.h>
#include <ie_ngraph_utils.hpp>
#include <utils/general_utils.h>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine::details;

namespace MKLDNNPlugin {

static InferenceEngine::LayerConfig make_plain_config(const std::shared_ptr<ngraph::Node>& op) {
    InferenceEngine::LayerConfig config;

    for (size_t i = 0; i < op->get_input_size(); i++) {
        const auto& dims = op->get_input_shape(i);
        const auto prec = InferenceEngine::details::convertPrecision(op->get_input_element_type(i));

        InferenceEngine::DataConfig data_conf {};
        data_conf.desc = InferenceEngine::TensorDesc { prec, dims, InferenceEngine::TensorDesc::getLayoutByDims(dims) };
        config.inConfs.push_back(data_conf);
    }

    for (size_t i = 0; i < op->get_output_size(); i++) {
        const auto& dims = op->get_output_shape(i);
        const auto prec = InferenceEngine::details::convertPrecision(op->get_output_element_type(i));

        InferenceEngine::DataConfig data_conf {};
        data_conf.desc = InferenceEngine::TensorDesc { prec, dims, InferenceEngine::TensorDesc::getLayoutByDims(dims) };
        config.outConfs.push_back(data_conf);
    }

    config.dynBatchSupport = true;
    return config;
}

class PortIteratorHelper : public PortMapHelper {
public:
    PortIteratorHelper(const MKLDNNMemoryPtr &from, const MKLDNNMemoryPtr &to, bool sliced_src,
                       const PortMap &slice_rule, const mkldnn::engine& eng)
                       : sliced_src(sliced_src) {
        const auto &full_blob = sliced_src ? from : to;
        const auto &part_blob = !sliced_src ? from : to;

        auto axis = slice_rule.axis;
        auto stride = slice_rule.stride;

        auto full_dims = full_blob->GetDims();
        auto part_dims = part_blob->GetDims();

        auto abs_stride = std::abs(stride);
        auto sign_of_stride = stride < 0.0f ? -1 : 1;

        iter_count = full_dims[axis] / abs_stride;

        full_dims[axis] = abs_stride;
        IE_ASSERT(full_dims == part_dims) << "Shape mismatch for tensor iterator port";

        // make chunk view
        auto chunk_desc =  full_blob->GetDescriptor();
        chunk_desc.data.dims[axis] = abs_stride;
        chunk_desc.data.padded_dims[axis] = abs_stride;  // TODO: asamption that plain tensor

        full_mem = full_blob->GetPrimitive();
        const auto full_mem_handler = full_mem.get_data_handle();
        mkldnn::memory chunk_mem = {chunk_desc, eng, full_mem_handler};

        auto elem_size = MKLDNNExtensionUtils::sizeOfDataType(mkldnn::memory::data_type(chunk_desc.data.data_type));

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

    void execute(mkldnn::stream strm, int iter) override {
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
    mkldnn::memory full_mem;

    int iter_count;
};

class BackEdgePortHelper : public PortMapHelper {
public:
    BackEdgePortHelper(const MKLDNNMemoryPtr &from, const MKLDNNMemoryPtr &to, const mkldnn::engine& eng) {
        mem_holder_src = from->GetPrimitive();
        mem_holder_dst = to->GetPrimitive();
        reorder = {mem_holder_src, mem_holder_dst};
    }

    void execute(mkldnn::stream strm, int iter) override {
        if (iter != 0) {
            reorder.execute(strm, mem_holder_src, mem_holder_dst);
        }
    }
};

class IterCountPortHelper : public PortMapHelper {
public:
    IterCountPortHelper(const MKLDNNMemoryPtr &to, const mkldnn::engine& eng) {
        // Only scalar I32 tensor is supported
        IE_ASSERT(to->GetDataType() == memory::data_type::s32);
        IE_ASSERT(to->GetDims() == memory::dims{1});
        mem_holder_dst = to->GetPrimitive();
    }

    void execute(mkldnn::stream strm, int n_iter) override {
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
    asBoolCheck(const MKLDNNMemoryPtr &mem) {
        IE_ASSERT(mem->GetDataType() == memory::data_type::u8);
        IE_ASSERT(mem->GetDims() == memory::dims{1});
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
    asIntCheck(const MKLDNNMemoryPtr &mem) {
        IE_ASSERT(mem->GetDataType() == memory::data_type::s32);
        IE_ASSERT(mem->GetDims() == memory::dims{1});
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

}  // namespace MKLDNNPlugin

int getNumIteration(const std::shared_ptr<const ngraph::Node>& op, const std::vector<PortMap>& inputPortMap, const std::vector<PortMap>& outputPortMap) {
    const auto isIterable = [](const PortMap& rule) { return rule.axis != -1; };

    const auto getNumIterations = [](const PortMap& rule, const std::vector<size_t>& dimensions) -> int {
        const auto axis = rule.axis;
        if (axis < 0 || static_cast<std::size_t>(axis) >= dimensions.size()) {
            IE_THROW() << R"(: Invalid "axis" value in an iteration component: )"
                               << rule.axis  << ", dimensions number = " << dimensions.size() << " (out of range)";
        }
        const auto space = dimensions[axis];
        const int start = static_cast<int>((rule.start < 0 ? (space + 1) : 0) + rule.start);
        const int end   = static_cast<int>((rule.end   < 0 ? (space + 1) : 0) + rule.end);

        const auto stride = rule.stride;
        if (stride == 0) {
            IE_THROW() << R"(: Invalid "stride" value in an iteration component: )" << rule.stride << " (infinite loop)";
        }
        const auto step = std::abs(stride);

        const auto src = stride < 0 ? end : start;
        const auto dst = stride < 0 ? start : end;
        const auto length = dst - src;
        if (src < 0 || src >= dst || dst > static_cast<int64_t>(space) || length < step) {
            IE_THROW() << R"(: Invalid "start"/"stride"/"end" values in an iteration component)"
                               << ": \"start\" = " << rule.start << ", \"stride\" = " << rule.stride  << ", \"end\" = " << rule.end;
        }

        if (length % step != 0) {
            IE_THROW() << ": Each iteration must be the same size: length (" << length << ") is not divisible by step (" << step << ")";
        }

        return static_cast<int>(length / step);
    };


    int numIterations = 1;
    bool isDefault = true;
    for (const auto& rule : inputPortMap) {
        if (!isIterable(rule)) {
            continue;
        }

        if (rule.from < 0 || rule.from >= static_cast<int64_t>(op->get_input_size())) {
            IE_THROW() << R"(: Invalid "from" value: "from" = )" << rule.from
                               << " inputs number = " << op->get_input_size() << " (out of range)";
        }

        const auto currentNumIterations = getNumIterations(rule, op->get_input_shape(rule.from));
        if (isDefault) {
            isDefault = false;
            numIterations = currentNumIterations;
        } else if (numIterations != currentNumIterations) {
            IE_THROW() << ": There are at least two different iterations numbers: " << numIterations << " and " << currentNumIterations;
        }
    }

    for (const auto& rule : outputPortMap) {
        if (!isIterable(rule)) {
            continue;
        }

        if (rule.from < 0 || rule.from >= static_cast<int64_t>(op->get_output_size())) {
            IE_THROW() << R"(: Invalid "from" value: "from" = )" << rule.from
                               << " inputs number = " << op->get_output_size() << " (out of range)";
        }

        const auto currentNumIterations = getNumIterations(rule, op->get_output_shape(rule.from));
        if (isDefault) {
            isDefault = false;
            numIterations = currentNumIterations;
        } else if (numIterations != currentNumIterations) {
            IE_THROW() << ": There are at least two different iterations numbers: " << numIterations << " and " << currentNumIterations;
        }
    }

    return numIterations;
}

bool MKLDNNTensorIteratorNode::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!one_of_castable(op->get_type_info(),
                ngraph::op::v0::TensorIterator::type_info,
                ngraph::op::v5::Loop::type_info)) {
            errorMessage = "Only opset1 TensorIterator or opset5 Loop operations are supported.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MKLDNNTensorIteratorNode::MKLDNNTensorIteratorNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache) :
        MKLDNNNode(op, eng, cache), ngraphOp(op) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }
}

void MKLDNNTensorIteratorNode::getSupportedDescriptors() {
    auto tiOp = std::dynamic_pointer_cast<ngraph::op::util::SubGraphOp>(ngraphOp);
    if (tiOp == nullptr) {
        IE_THROW() << "Can't cast TensorIterator node with name: " << getName() << " to ngraph::op::util::SubGraphOp";
    }
    const std::shared_ptr<const ngraph::Function> body = tiOp->get_function();
    sub_graph.CreateGraph(body, ext_mng, weightCache);

    const auto &inMap = sub_graph.GetInputNodesMap();
    for (const auto &param : tiOp->get_function()->get_parameters()) {
        auto inNode = inMap.find(param->get_friendly_name());
        if (inNode != inMap.end()) {
            auto inMem = inNode->second->getChildEdgeAt(0)->getMemoryPtr();
            input_mem.push_back(inMem);
        }
    }

    const auto &outMap = sub_graph.GetOutputNodesMap();
    for (const auto &out : tiOp->get_function()->get_results()) {
        auto prev = out->get_input_node_shared_ptr(0);
        std::string inputID = prev->get_friendly_name();
        if (prev->get_output_size() > 1) {
            inputID += "." + std::to_string(out->get_input_source_output(0).get_index());
        }
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
            auto output_desc = ::ngraph::as_type_ptr<ngraph::op::util::SubGraphOp::ConcatOutputDescription>(desc);
            IE_ASSERT(output_desc != nullptr);

            outputPortMap.emplace_back(PortMap {
                static_cast<int>(output_desc->m_output_index), static_cast<int>(body_output_idx),
                static_cast<int>(output_desc->m_axis), static_cast<int>(output_desc->m_stride),
                static_cast<int>(output_desc->m_start), static_cast<int>(output_desc->m_end),
                static_cast<int>(output_desc->m_part_size)});
        } else if (type_name == "BodyOutputDescription") {
            auto output_desc = ::ngraph::as_type_ptr<ngraph::op::util::SubGraphOp::BodyOutputDescription>(desc);
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

        if (const auto slice_desc = std::dynamic_pointer_cast<ngraph::op::util::SubGraphOp::SliceInputDescription>(desc)) {
            inputPortMap.emplace_back(PortMap {
                static_cast<int>(slice_desc->m_input_index), static_cast<int>(body_input_index),
                static_cast<int>(slice_desc->m_axis), static_cast<int>(slice_desc->m_stride),
                static_cast<int>(slice_desc->m_start), static_cast<int>(slice_desc->m_end),
                static_cast<int>(slice_desc->m_part_size)});
        } else if (const auto merge_desc = std::dynamic_pointer_cast<ngraph::op::util::SubGraphOp::MergedInputDescription>(desc)) {
            inputPortMap.emplace_back(PortMap {
                static_cast<int>(merge_desc->m_input_index), static_cast<int>(body_input_index), -1, 1, 0, -1, 1});

            auto body_output_idx = merge_desc->m_body_value_index;

            backEdges.emplace_back(PortMap {
                static_cast<int>(body_output_idx), static_cast<int>(body_input_index), -1, 1, 0, -1, 1});
        } else if (const auto inv_desc = std::dynamic_pointer_cast<ngraph::op::util::SubGraphOp::InvariantInputDescription>(desc)) {
            inputPortMap.emplace_back(PortMap {
                    static_cast<int>(inv_desc->m_input_index), static_cast<int>(body_input_index), -1, 1, 0, -1, 1});
        } else {
            IE_THROW() << "Incorrect type of the input description.";
        }
    }

    n_iter = getNumIteration(ngraphOp, inputPortMap, outputPortMap);

    if (const auto loopOp = std::dynamic_pointer_cast<const ngraph::op::v5::Loop>(ngraphOp)) {
        auto spec_port = loopOp->get_special_body_ports();
        if (spec_port.current_iteration_input_idx != -1) {
            loopBodyCurrentIterationIdx.push_back(spec_port.current_iteration_input_idx);
        }
        if (spec_port.body_condition_output_idx != -1) {
            loopBodyConditionOutputIdx = spec_port.body_condition_output_idx;
        }
        loopTripCountIdx = 0;
        loopExecutionConditionIdx = 1;
    }

    config = make_plain_config(ngraphOp);
}

void MKLDNNTensorIteratorNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown);
}


void MKLDNNTensorIteratorNode::createPrimitive() {
    const auto &eng = getEngine();

    for (auto map_rule : inputPortMap) {
        auto &from_mem = getParentEdgesAtPort(map_rule.from)[0]->getMemoryPtr();
        auto &to_mem = input_mem[map_rule.to];

        if (map_rule.axis == -1)
            first_mappers.emplace_back(new BackEdgePortHelper(from_mem, to_mem, eng));
        else
            before_mappers.emplace_back(new PortIteratorHelper(from_mem, to_mem, true, map_rule, eng));
    }

    for (auto map_rule : outputPortMap) {
        auto &to_mem = getChildEdgesAtPort(map_rule.from)[0]->getMemoryPtr();
        auto &from_mem = output_mem[map_rule.to];

        if (map_rule.axis == -1)
            last_mappers.emplace_back(new BackEdgePortHelper(from_mem, to_mem, eng));
        else
            after_mappers.emplace_back(new PortIteratorHelper(from_mem, to_mem, false, map_rule, eng));
    }

    for (auto map_rule : backEdges) {
        auto from_mem = output_mem[map_rule.from];
        auto to_mem = input_mem[map_rule.to];

        before_mappers.emplace_back(new BackEdgePortHelper(from_mem, to_mem, eng));
    }

    // special purpose ports
    for (auto idx : loopBodyCurrentIterationIdx) {
        auto to_mem = input_mem[idx];
        before_mappers.emplace_back(new IterCountPortHelper(to_mem, eng));
    }

    if (loopBodyConditionOutputIdx == -1) {
        continue_cond_check.reset(new staticValueCheck(true)); // always true
    } else {
        auto mem = output_mem[loopBodyConditionOutputIdx];
        continue_cond_check.reset(new asBoolCheck(mem));
    }

    if (loopTripCountIdx == -1) {
        trip_count_check.reset(new staticValueCheck(n_iter)); // use statically calculated num of iteration
    } else {
        auto mem = getParentEdgesAtPort(loopTripCountIdx)[0]->getMemoryPtr();
        trip_count_check.reset(new asIntCheck(mem));
    }

    if (loopExecutionConditionIdx == -1) {
        initial_cond_check.reset(new staticValueCheck(true));
    } else {
        auto mem = getParentEdgesAtPort(loopExecutionConditionIdx)[0]->getMemoryPtr();
        initial_cond_check.reset(new asBoolCheck(mem));
    }
}

void MKLDNNTensorIteratorNode::execute(mkldnn::stream strm) {
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
        // or to next iteration inputs
        for (auto &mapper : after_mappers)
            mapper->execute(strm, i);
    }

    for (auto &mapper : last_mappers)
        mapper->execute(strm);
}

bool MKLDNNTensorIteratorNode::created() const {
    return getType() == TensorIterator;
}
REG_MKLDNN_PRIM_FOR(MKLDNNTensorIteratorNode, TensorIterator);
