// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "base.hpp"
#include <ngraph/op/gather_tree.hpp>
#include <nodes/common/tensor_desc_creator.h>
#include <utils/general_utils.h>

#include <cmath>
#include <limits>
#include <cfloat>
#include <string>
#include <vector>
#include <cassert>
#include <algorithm>
#include "ie_parallel.hpp"

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

using MKLDNNPlugin::TensorDescCreatorTypes;

class GatherTreeImpl: public ExtLayerBase {
public:
    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
        try {
            auto gatherElementsOp = ngraph::as_type_ptr<const ngraph::op::v1::GatherTree>(op);
            if (!gatherElementsOp) {
                errorMessage = "Node is not an instance of the GatherTree operation from operation set v1.";
                return false;
            }

            auto precision = op->get_input_element_type(GATHER_TREE_STEP_IDX);
            if (!MKLDNNPlugin::one_of(precision, ngraph::element::f32, ngraph::element::i32))
                precision = ngraph::element::f32;
            if (op->get_input_element_type(GATHER_TREE_PARENT_IDX) != precision ||
                    op->get_input_element_type(GATHER_TREE_MAX_SEQ_LEN) != precision ||
                    op->get_input_element_type(GATHER_TREE_END_TOKEN) != precision ||
                    op->get_output_element_type(0) != precision) {
                errorMessage = "Node has incorrect input/output data precision. Must be the same.";
                return false;
            }
        } catch (...) {
            return false;
        }

        return true;
    }

    explicit GatherTreeImpl(const std::shared_ptr<ngraph::Node>& op) {
        try {
            std::string errorMessage;
            if (!isSupportedOperation(op, errorMessage)) {
                IE_THROW(NotImplemented) << errorMessage;
            }

            std::string errorPrefix = std::string("Node GatherTree with name '") + op->get_friendly_name() + "'";
            if (op->get_input_size() != 4)
                IE_THROW() << errorPrefix << " has incorrect number of input edges.";
            if (op->get_output_size() != 1)
                IE_THROW() << errorPrefix << " has incorrect number of output edges.";

            precision = details::convertPrecision(op->get_input_element_type(GATHER_TREE_STEP_IDX));
            if (!MKLDNNPlugin::one_of(precision, Precision::FP32, Precision::I32))
                precision = Precision::FP32;

            if (op->get_input_shape(GATHER_TREE_STEP_IDX).size() != 3)
                IE_THROW() << errorPrefix << " step_idx vector should be 3 dimension";
            if (op->get_input_shape(GATHER_TREE_PARENT_IDX).size() != 3)
                IE_THROW() << errorPrefix << " parent_idx vector should be 3 dimension";
            if (op->get_input_shape(GATHER_TREE_MAX_SEQ_LEN).size() != 1)
                IE_THROW() << errorPrefix << " max_seq_len vector should be 1 dimension";
            if (op->get_input_shape(GATHER_TREE_END_TOKEN).size() != 0)
                IE_THROW() << errorPrefix << " end_token should be 1 dimension";

            addConfig(op, {{TensorDescCreatorTypes::ncsp, precision},
                           {TensorDescCreatorTypes::ncsp, precision},
                           {TensorDescCreatorTypes::ncsp, precision},
                           {TensorDescCreatorTypes::ncsp, precision}},
                          {{TensorDescCreatorTypes::ncsp, precision}});
        } catch (InferenceEngine::Exception &ex) {
            errorMsg = ex.what();
            throw;
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs, ResponseDesc *resp) noexcept override {
        if (precision == Precision::FP32)
            return execute_impl<float  >(inputs, outputs, resp);
        else
            return execute_impl<int32_t>(inputs, outputs, resp);
    }

    template<typename DATA_T>
    StatusCode execute_impl(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs, ResponseDesc *resp) noexcept {
        const auto *step_idx = inputs[GATHER_TREE_STEP_IDX]->cbuffer().as<DATA_T *>() +
            inputs[GATHER_TREE_STEP_IDX]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        const auto * const parent_idx = inputs[GATHER_TREE_PARENT_IDX]->cbuffer().as<DATA_T *>() +
            inputs[GATHER_TREE_PARENT_IDX]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        const size_t parent_idx_size = inputs[GATHER_TREE_PARENT_IDX]->size()
            - inputs[GATHER_TREE_PARENT_IDX]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        const auto *max_seq_len = inputs[GATHER_TREE_MAX_SEQ_LEN]->cbuffer().as<DATA_T *>() +
            inputs[GATHER_TREE_MAX_SEQ_LEN]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        auto end_token = (inputs[GATHER_TREE_END_TOKEN]->cbuffer().as<DATA_T *>() +
            inputs[GATHER_TREE_END_TOKEN]->getTensorDesc().getBlockingDesc().getOffsetPadding())[0];
        auto * final_idx = outputs[0]->cbuffer().as<DATA_T *>() +
            outputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();

        SizeVector step_idx_dims = inputs[GATHER_TREE_STEP_IDX]->getTensorDesc().getDims();
        SizeVector parent_idx_dims = inputs[GATHER_TREE_PARENT_IDX]->getTensorDesc().getDims();
        SizeVector max_seq_len_dims = inputs[GATHER_TREE_MAX_SEQ_LEN]->getTensorDesc().getDims();
        SizeVector final_idx_dims = outputs[0]->getTensorDesc().getDims();
        int32_t max_time = step_idx_dims[0];
        const size_t batch_size = step_idx_dims[1];
        const size_t beam_width = step_idx_dims[2];
        const size_t bb_size = batch_size * beam_width;

        if (max_time != static_cast<int32_t>(parent_idx_dims[0]) || max_time != static_cast<int32_t>(final_idx_dims[0]) ||
            batch_size != parent_idx_dims[1] || batch_size != final_idx_dims[1] || batch_size != max_seq_len_dims[0] ||
            beam_width != parent_idx_dims[2] || beam_width != final_idx_dims[2]) {
            if (resp) {
                std::string errorMsg = "Input/Output tensors dimensions mismatch";
                errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
            }
            return PARAMETER_MISMATCH;
        }

        bool incorrect_result = false;
        parallel_for2d(batch_size, beam_width, [&](size_t batch, size_t beam) {
            int32_t max_sequence_in_beam = std::min<int32_t>(max_time, static_cast<int32_t>(max_seq_len[batch]));
            if (max_sequence_in_beam > 0) {
                int32_t time, idx = (max_time - 1) * bb_size + batch * beam_width;
                for (time = (max_time - 1); time >= max_sequence_in_beam; time--, idx -= bb_size)
                    final_idx[idx + beam] = end_token;

                for (int32_t parent = static_cast<int32_t>(beam); time >= 0; time--, idx -= bb_size) {
                    if (parent < 0
                            || parent >= static_cast<int32_t>(beam_width)
                            || idx + parent >= parent_idx_size) {
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
            if (resp) {
                std::string errorMsg = "Wrong parent index, result is incorrect";
                errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
            }
            return OUT_OF_BOUNDS;
        }

        return OK;
    }

private:
    static const size_t GATHER_TREE_STEP_IDX = 0;
    static const size_t GATHER_TREE_PARENT_IDX = 1;
    static const size_t GATHER_TREE_MAX_SEQ_LEN = 2;
    static const size_t GATHER_TREE_END_TOKEN = 3;

    InferenceEngine::Precision precision;
};

REG_FACTORY_FOR(GatherTreeImpl, GatherTree);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
