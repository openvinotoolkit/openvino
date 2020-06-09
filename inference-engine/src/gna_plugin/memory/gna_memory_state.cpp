// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gna_memory_state.hpp"
#include "frontend/quantized_layer_params.hpp"
#include "layer_transform.hpp"
#include "preprocessing.hpp"
#include "ie_layouts.h"

namespace  GNAPluginNS {

namespace memory {

    std::string GNAMemoryState::GetName() const {
        return name;
    }

    void GNAMemoryState::Reset() {
        state->Reset();
    }

    InferenceEngine::Precision getPrecision(const GNAMemoryLayer* state) {
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
                THROW_GNA_EXCEPTION << "Incorrect state element size to determine precision";
            }
        }

        return state_precision;
    }

    void GNAMemoryState::SetState(InferenceEngine::Blob::Ptr newState) {
        auto data_ptr = newState->cbuffer().as<void*>();
        auto data_size = newState->byteSize();
        auto data_elements = data_size / newState->element_size();
        if (ALIGN64(state->reserved_size) != ALIGN64((data_size / (newState->element_size() / state->elementSizeBytes())))) {
            THROW_GNA_EXCEPTION << "Failed to SetState. Sizes of new and old states do not match";
        }

        InferenceEngine::Precision state_precision = getPrecision(state);
        auto new_state_precision = newState->getTensorDesc().getPrecision();

        if (new_state_precision == state_precision) {
            std::memcpy(state->gna_ptr, data_ptr, data_size);
            return;
        }

        switch (state_precision) {
        case InferenceEngine::Precision::I16: {
            if (new_state_precision == InferenceEngine::Precision::FP32) {
                auto quantized = InferenceEngine::getInjectedData<QuantizedLayerParams>(state->getInput());
                auto scale_factor = quantized != nullptr ? quantized->_dst_quant.scale : 1.0f;
                GNAPluginNS::ConvertToInt16(static_cast<int16_t*>(state->gna_ptr),
                    newState->buffer().as<float*>(),
                    1,
                    data_elements,
                    scale_factor);
            } else {
                THROW_GNA_EXCEPTION << "Failed to SetState. If old state precision is I16 only I16 and FP32 are allowed as new state precisions.";
            }
            break;
        }
        default:
            THROW_GNA_EXCEPTION << "Failed to SetState. Incorrect new/old precision pair";
        }
    }

    InferenceEngine::Blob::CPtr GNAMemoryState::GetLastState() const {
        auto elements = state->reserved_size / state->elementSizeBytes();
        InferenceEngine::Precision state_precision = getPrecision(state);
        auto result_blob = make_blob_with_precision(InferenceEngine::TensorDesc(state_precision,
            InferenceEngine::SizeVector({1, elements}),
            InferenceEngine::NC),
            state->gna_ptr);

        return result_blob;
    }
}  // namespace memory
}  // namespace GNAPluginNS
