// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "list.hpp"
#include "base.hpp"

#include <vector>

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class OneHotImpl: public ExtLayerBase {
public:
    explicit OneHotImpl(const CNNLayer* layer) {
        try {
            depth     = layer->GetParamAsUInt("depth");
            on_value  = layer->GetParamAsFloat("on_value", 1.0f);
            off_value = layer->GetParamAsFloat("off_value", 0.0f);
            axis      = layer->GetParamAsInt("axis", -1);

            src_dims = layer->insData[0].lock()->getTensorDesc().getDims();
            dst_dims = layer->outData[0]->getTensorDesc().getDims();

            int output_dims_size = dst_dims.size();
            if (layer->CheckParamPresence("axis") &&
                (-1 > axis || axis >= output_dims_size)) {
                    THROW_IE_EXCEPTION << "The value of " << layer->name << " layer axis parameter must be between -1 <= axis < "\
                                       << output_dims_size << ", but actually it is " << axis;
            }

            if (!( ((1 + src_dims.size()) == dst_dims.size()) ||
                   (src_dims.size() == 1 && dst_dims.size() == 1 && dst_dims[0] == depth && src_dims[0] == 1)))
                THROW_IE_EXCEPTION << layer->name << " Incorrect number of input/output dimensions!";

            // check a precision of the input tensor
            input_precision = layer->insData[0].lock()->getTensorDesc().getPrecision();
            if (input_precision != Precision::I32 && input_precision != Precision::FP32) {
                THROW_IE_EXCEPTION << layer->name << " Incorrect input precision for the input. Only I32 and FP32 are supported!";
            }

            addConfig(layer, { DataConfigurator(ConfLayout::PLN) }, { DataConfigurator(ConfLayout::PLN) });
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs, ResponseDesc *resp) noexcept override {
        switch (input_precision) {
            case Precision::FP32:
                one_hot<float>(inputs[0], outputs[0]);
                break;
            case Precision::I32:
                one_hot<int>(inputs[0], outputs[0]);
                break;
            default:
                return GENERAL_ERROR;
        }

        return OK;
    }

private:
    template <typename in_type>
    void one_hot(Blob::Ptr input, Blob::Ptr output) {
        const auto *src_data = input->cbuffer().as<const in_type *>();
        auto *dst_data = output->buffer().as<float *>();
        std::size_t prefix_size = 1;
        auto input_dims = input->getTensorDesc().getDims();

        std::size_t actual_axis = (axis == -1) ? src_dims.size() : axis;
        for (size_t i = 0; i < actual_axis; ++i)
            prefix_size *= input_dims[i];

        std::size_t suffix_size = input->size() / prefix_size;

        std::size_t dst_offset = 0;
        for (std::size_t prefix_idx = 0; prefix_idx < prefix_size; ++prefix_idx) {
            for (std::size_t depth_idx = 0; depth_idx < depth; ++depth_idx) {
                for (std::size_t suffix_idx = 0; suffix_idx < suffix_size; suffix_idx++) {
                    auto src_index = prefix_idx * suffix_size + suffix_idx;
                    std::size_t v = static_cast<std::size_t>(src_data[src_index]);
                    dst_data[dst_offset++] = (v == depth_idx) ? on_value : off_value;
                }
            }
        }
    }

    uint32_t depth;
    float on_value = 1.f;
    float off_value = 0.f;
    int32_t axis = -1;
    SizeVector src_dims;
    SizeVector dst_dims;

    Precision input_precision;
};

REG_FACTORY_FOR(ImplFactory<OneHotImpl>, OneHot);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
