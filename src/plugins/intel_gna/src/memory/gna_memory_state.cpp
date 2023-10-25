// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gna_memory_state.hpp"

#include "frontend/quantized_layer_params.hpp"
#include "gna_graph_tools.hpp"
#include "ie_layouts.h"
#include "pre_post_process/data_conversion_helpers.hpp"

namespace ov {
namespace intel_gna {
namespace memory {

void GNAVariableState::Reset() {
    state->Reset();
}

InferenceEngine::Precision GNAVariableState::getPrecision() const {
    InferenceEngine::Precision state_precision;

    if (state->getInput()) {
        state_precision = state->getInput()->precision;
    } else {
        auto element_size = state->elementSizeBytes();
        switch (element_size) {
        case 4:
            state_precision = InferenceEngine::Precision::FP32;
            break;
        case 2:
            state_precision = InferenceEngine::Precision::I16;
            break;
        default:
            THROW_GNA_EXCEPTION << "Incorrect state element size " << element_size
                                << " to determine precision for VariableState " << name;
        }
    }

    return state_precision;
}

void GNAVariableState::SetState(const InferenceEngine::Blob::Ptr& newState) {
    IE_ASSERT(newState != nullptr);

    auto data_ptr = newState->cbuffer().as<void*>();
    IE_ASSERT(data_ptr != nullptr);
    auto data_size = newState->byteSize();
    auto data_elements = data_size / newState->element_size();
    if (state->reserved_size > (data_size / (newState->element_size() / state->elementSizeBytes()))) {
        THROW_GNA_EXCEPTION << "Failed to SetState. Sizes of new and old states do not match. (" << state->reserved_size
                            << " != " << (newState->element_size() / state->elementSizeBytes()) << ")";
    }

    InferenceEngine::Precision state_precision = getPrecision();
    auto new_state_precision = newState->getTensorDesc().getPrecision();

    if (state->gna_ptr == data_ptr) {
        return;
    }

    if (new_state_precision == state_precision) {
        std::memcpy(state->gna_ptr, data_ptr, data_size);
        return;
    }

    switch (state_precision) {
    case InferenceEngine::Precision::I16: {
        if (new_state_precision == InferenceEngine::Precision::FP32) {
            auto quantized =
                InferenceEngine::getInjectedData<ov::intel_gna::frontend::QuantizedLayerParams>(state->getInput());
            auto scale_factor = quantized != nullptr ? quantized->_dst_quant.GetScale() : state->scale_factor;
            pre_post_processing::ConvertToInt16(static_cast<int16_t*>(state->gna_ptr),
                                                newState->buffer().as<float*>(),
                                                1,
                                                static_cast<uint32_t>(data_elements),
                                                scale_factor);
        } else {
            THROW_GNA_EXCEPTION
                << "Failed to SetState for VariableState " << name
                << ". If old state precision is I16 only I16 and FP32 are allowed as new state precisions."
                << " Old state: " << state_precision << " New state: " << new_state_precision;
        }
        break;
    }
    default:
        THROW_GNA_EXCEPTION << "Failed to SetState for VariableState " << name << ". Incorrect new/old precision pair"
                            << " Old state: " << state_precision << " New state: " << new_state_precision;
    }
}

InferenceEngine::Blob::CPtr GNAVariableState::GetState() const {
    auto elements = state->reserved_size / state->elementSizeBytes();
    InferenceEngine::Precision state_precision = getPrecision();

    if (state->getInput() && state_precision == InferenceEngine::Precision::I16) {
        auto quantized =
            InferenceEngine::getInjectedData<ov::intel_gna::frontend::QuantizedLayerParams>(state->getInput());
        auto scale_factor = quantized != nullptr ? quantized->_dst_quant.GetScale() : state->scale_factor;

        auto result_blob =
            make_blob_with_precision(InferenceEngine::TensorDesc(InferenceEngine::Precision::FP32,
                                                                 InferenceEngine::SizeVector({1, elements}),
                                                                 InferenceEngine::NC));

        result_blob->allocate();
        auto buffer = result_blob->buffer().as<float*>();
        auto new_gna_ptr = static_cast<int16_t*>(state->gna_ptr);

        for (size_t i = 0; i < elements; i++) {
            buffer[i] = new_gna_ptr[i] / scale_factor;
        }

        return result_blob;
    } else {
        auto result_blob =
            make_blob_with_precision(InferenceEngine::TensorDesc(state_precision,
                                                                 InferenceEngine::SizeVector({1, elements}),
                                                                 InferenceEngine::NC));
        result_blob->allocate();
        std::memcpy(result_blob->buffer(), state->gna_ptr, state->reserved_size);
        return result_blob;
    }
}

float GNAVariableState::GetScaleFactor() const {
    auto quantized = InferenceEngine::getInjectedData<ov::intel_gna::frontend::QuantizedLayerParams>(state->getInput());
    auto scale_factor = quantized != nullptr ? quantized->_dst_quant.GetScale() : state->scale_factor;
    return scale_factor;
}

}  // namespace memory
}  // namespace intel_gna
}  // namespace ov
