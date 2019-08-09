// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ext_list.hpp"
#include "ext_base.hpp"

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

class GatherTreeImpl: public ExtLayerBase {
public:
    explicit GatherTreeImpl(const CNNLayer* layer) {
        try {
            if (layer->insData.empty() || layer->outData.empty())
                THROW_IE_EXCEPTION << layer->name << " Incorrect number of input/output edges.";

            if (layer->insData.size() != 4)
                THROW_IE_EXCEPTION << layer->name << " Incorrect number of input edges.";
            if (layer->outData.size() != 1)
                THROW_IE_EXCEPTION << layer->name << " Incorrect number of output edges.";

            precision = layer->insData[GATHER_TREE_STEP_IDX].lock()->getTensorDesc().getPrecision();

            if (precision != Precision::FP32 && precision != Precision::I32)
                THROW_IE_EXCEPTION << layer->name << " Incorrect data tensor precision. Only I32 or FP32 are supported.";

            if (layer->insData[GATHER_TREE_PARENT_IDX].lock()->getTensorDesc().getPrecision() != precision ||
                layer->insData[GATHER_TREE_MAX_SEQ_LEN].lock()->getTensorDesc().getPrecision() != precision ||
                layer->insData[GATHER_TREE_END_TOKEN].lock()->getTensorDesc().getPrecision() != precision ||
                layer->outData[0]->getTensorDesc().getPrecision() != precision)
                THROW_IE_EXCEPTION << layer->name << " Incorrect input/output data tensor precision. Should be the same.";

            if (layer->insData[GATHER_TREE_STEP_IDX].lock()->getTensorDesc().getDims().size() != 3)
                THROW_IE_EXCEPTION << layer->name << " step_idx vector should be 3 dimension";
            if (layer->insData[GATHER_TREE_PARENT_IDX].lock()->getTensorDesc().getDims().size() != 3)
                THROW_IE_EXCEPTION << layer->name << " parent_idx vector should be 3 dimension";
            if (layer->insData[GATHER_TREE_MAX_SEQ_LEN].lock()->getTensorDesc().getDims().size() != 1)
                THROW_IE_EXCEPTION << layer->name << " max_seq_len vector should be 1 dimension";
            if (layer->insData[GATHER_TREE_END_TOKEN].lock()->getTensorDesc().getDims().size() != 1)
                THROW_IE_EXCEPTION << layer->name << " end_token should be 1 dimension";

            addConfig(layer, { DataConfigurator(ConfLayout::PLN), DataConfigurator(ConfLayout::PLN),
                               DataConfigurator(ConfLayout::PLN), DataConfigurator(ConfLayout::PLN) },
                             { DataConfigurator(ConfLayout::PLN) });
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
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
        const auto *parent_idx = inputs[GATHER_TREE_PARENT_IDX]->cbuffer().as<DATA_T *>() +
            inputs[GATHER_TREE_PARENT_IDX]->getTensorDesc().getBlockingDesc().getOffsetPadding();
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
        size_t batch_size = step_idx_dims[1];
        size_t beam_width = step_idx_dims[2];
        size_t bb_size = batch_size * beam_width;

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
                    if (parent < 0 || parent >= static_cast<int32_t>(beam_width)) {
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
    const size_t GATHER_TREE_STEP_IDX = 0;
    const size_t GATHER_TREE_PARENT_IDX = 1;
    const size_t GATHER_TREE_MAX_SEQ_LEN = 2;
    const size_t GATHER_TREE_END_TOKEN = 3;

    InferenceEngine::Precision precision;
};

REG_FACTORY_FOR(ImplFactory<GatherTreeImpl>, GatherTree);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
