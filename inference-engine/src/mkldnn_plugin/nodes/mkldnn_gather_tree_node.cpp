// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>
#include <cmath>

#include <ngraph/op/gather_tree.hpp>
#include "ie_parallel.hpp"
#include "mkldnn_gather_tree_node.h"
#include <utils/general_utils.h>

using namespace MKLDNNPlugin;
using namespace InferenceEngine;

bool MKLDNNGatherTreeNode::isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto gatherElementsOp = ngraph::as_type_ptr<const ngraph::op::v1::GatherTree>(op);
        if (!gatherElementsOp) {
            errorMessage = "Node is not an instance of the GatherTree operation from operation set v1.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MKLDNNGatherTreeNode::MKLDNNGatherTreeNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng,
        MKLDNNWeightsSharing::Ptr &cache) : MKLDNNNode(op, eng, cache) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    errorPrefix = std::string("Node GatherTree with name '") + op->get_friendly_name() + "'";
    if (op->get_input_size() != 4)
        IE_THROW() << errorPrefix << " has incorrect number of input edges.";
    if (op->get_output_size() != 1)
        IE_THROW() << errorPrefix << " has incorrect number of output edges.";

    if (op->get_input_shape(GATHER_TREE_STEP_IDX).size() != 3)
        IE_THROW() << errorPrefix << " step_idx vector should be 3 dimension";
    if (op->get_input_shape(GATHER_TREE_PARENT_IDX).size() != 3)
        IE_THROW() << errorPrefix << " parent_idx vector should be 3 dimension";
    if (op->get_input_shape(GATHER_TREE_MAX_SEQ_LEN).size() != 1)
        IE_THROW() << errorPrefix << " max_seq_len vector should be 1 dimension";
    if (op->get_input_shape(GATHER_TREE_END_TOKEN).size() != 0)
        IE_THROW() << errorPrefix << " end_token should be 1 dimension";
}

void MKLDNNGatherTreeNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    precision = getOriginalInputPrecisionAtPort(GATHER_TREE_STEP_IDX);
    if (!MKLDNNPlugin::one_of(precision, Precision::FP32, Precision::I32))
        precision = Precision::FP32;

    if (getOriginalInputPrecisionAtPort(GATHER_TREE_PARENT_IDX)  != precision ||
        getOriginalInputPrecisionAtPort(GATHER_TREE_MAX_SEQ_LEN) != precision ||
        getOriginalInputPrecisionAtPort(GATHER_TREE_END_TOKEN)   != precision ||
        getOriginalOutputPrecisionAtPort(0)                 != precision) {
            IE_THROW() << errorPrefix << " has incorrect input/output data precision. Must be the same.";
    }

    addSupportedPrimDesc({{GeneralLayout::ncsp, precision},
                            {GeneralLayout::ncsp, precision},
                            {GeneralLayout::ncsp, precision},
                            {GeneralLayout::ncsp, precision}},
                         {{GeneralLayout::ncsp, precision}},
                         impl_desc_type::ref_any);
}

void MKLDNNGatherTreeNode::execute(mkldnn::stream strm) {
    if (precision == Precision::FP32)
        return gatherTreeKernel<float>();
    else
        return gatherTreeKernel<int32_t>();
}

template<typename DATA_T>
void MKLDNNGatherTreeNode::gatherTreeKernel() noexcept {
    const auto *step_idx = reinterpret_cast<DATA_T *>(getParentEdgeAt(GATHER_TREE_STEP_IDX)->getMemoryPtr()->GetPtr());
    const auto * const parent_idx = reinterpret_cast<DATA_T *>(getParentEdgeAt(GATHER_TREE_PARENT_IDX)->getMemoryPtr()->GetPtr());
    const size_t parent_idx_size = getParentEdgeAt(GATHER_TREE_PARENT_IDX)->getShape().getElementsCount()
                                   - getParentEdgeAt(GATHER_TREE_PARENT_IDX)->getMemory().GetDescWithType<BlockedMemoryDesc>().getOffsetPadding();
    const auto *max_seq_len = reinterpret_cast<DATA_T *>(getParentEdgeAt(GATHER_TREE_MAX_SEQ_LEN)->getMemoryPtr()->GetPtr());
    auto end_token = (reinterpret_cast<DATA_T *>(getParentEdgeAt(GATHER_TREE_END_TOKEN)->getMemoryPtr()->GetPtr()))[0];
    auto * final_idx = reinterpret_cast<DATA_T *>(getChildEdgesAtPort(0)[0]->getMemoryPtr()->GetPtr());

    SizeVector step_idx_dims = getParentEdgeAt(GATHER_TREE_STEP_IDX)->getShape().getStaticDims();
    SizeVector parent_idx_dims = getParentEdgeAt(GATHER_TREE_PARENT_IDX)->getShape().getStaticDims();
    SizeVector max_seq_len_dims = getParentEdgeAt(GATHER_TREE_MAX_SEQ_LEN)->getShape().getStaticDims();
    SizeVector final_idx_dims = getChildEdgesAtPort(0)[0]->getShape().getStaticDims();
    int32_t max_time = step_idx_dims[0];
    const size_t batch_size = step_idx_dims[1];
    const size_t beam_width = step_idx_dims[2];
    const size_t bb_size = batch_size * beam_width;

    if (max_time != static_cast<int32_t>(parent_idx_dims[0]) || max_time != static_cast<int32_t>(final_idx_dims[0]) ||
        batch_size != parent_idx_dims[1] || batch_size != final_idx_dims[1] || batch_size != max_seq_len_dims[0] ||
        beam_width != parent_idx_dims[2] || beam_width != final_idx_dims[2]) {
        std::string errorMsg = "Input/Output tensors dimensions mismatch";
        IE_THROW() << errorMsg;
    }

    bool incorrect_result = false;
    parallel_for2d(batch_size, beam_width, [&](size_t batch, size_t beam) {
        int32_t max_sequence_in_beam = std::min<int32_t>(max_time, static_cast<int32_t>(max_seq_len[batch]));
        if (max_sequence_in_beam > 0) {
            int32_t time, idx = (max_time - 1) * bb_size + batch * beam_width;
            for (time = (max_time - 1); time >= max_sequence_in_beam; time--, idx -= bb_size)
                final_idx[idx + beam] = end_token;

            for (int32_t parent = static_cast<int32_t>(beam); time >= 0; time--, idx -= bb_size) {
                if (parent < 0 || parent >= static_cast<int32_t>(beam_width) || idx + parent >= parent_idx_size) {
                    incorrect_result = true;
                    break;
                }
                final_idx[idx + beam] = step_idx[idx + parent];
                parent = static_cast<int32_t>(parent_idx[idx + parent]);
            }

            bool finished = false;
            auto *final = &final_idx[batch * beam_width + beam];
            for (time = 0; time < max_sequence_in_beam; time++, final += bb_size) {
                if (finished)
                    (*final) = end_token;
                else if ((*final) == end_token)
                    finished = true;
            }
        }
    });

    if (incorrect_result) {
        std::string errorMsg = "Wrong parent index, result is incorrect";
        IE_THROW() << errorMsg;
    }
}

bool MKLDNNGatherTreeNode::created() const {
    return getType() == GatherTree;
}

REG_MKLDNN_PRIM_FOR(MKLDNNGatherTreeNode, GatherTree)
