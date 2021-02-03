// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "base.hpp"
#include "ie_parallel.hpp"
#include "common/utils.hpp"
#include "utils/bfloat16.hpp"
#include <mkldnn_selective_build.h>

#include <vector>

using namespace dnnl::impl::utils;

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
            if (!one_of(input_precision, Precision::I32, Precision::FP32, Precision::BF16)) {
                THROW_IE_EXCEPTION << layer->name << " Incorrect input precision for the input. Only I32, FP32 and BF16 are supported!";
            }
            output_precision = layer->outData[0]->getTensorDesc().getPrecision();
            if (!one_of(output_precision, Precision::FP32, Precision::BF16)) {
                THROW_IE_EXCEPTION << layer->name << " Incorrect precision for the output. Only FP32 and BF16 are supported!";
            }

            addConfig(layer, { DataConfigurator(ConfLayout::PLN, input_precision) }, { DataConfigurator(ConfLayout::PLN, output_precision) });
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs, ResponseDesc *resp) noexcept override {
        OneHotContext ctx = {this, inputs[0], outputs[0], false};
        OV_SWITCH(MKLDNNPlugin, OneHotExecute, ctx, std::tie(input_precision, output_precision),
                  OV_CASE2(Precision::FP32, Precision::FP32, float, float),
                  OV_CASE2(Precision::I32, Precision::FP32, int, float),
                  OV_CASE2(Precision::BF16, Precision::FP32, MKLDNNPlugin::bfloat16_t, float),
                  OV_CASE2(Precision::FP32, Precision::BF16, float, MKLDNNPlugin::bfloat16_t),
                  OV_CASE2(Precision::I32, Precision::BF16, int, MKLDNNPlugin::bfloat16_t),
                  OV_CASE2(Precision::BF16, Precision::BF16, MKLDNNPlugin::bfloat16_t, MKLDNNPlugin::bfloat16_t))

        if (!ctx.executed) {
            return GENERAL_ERROR;
        }
        return OK;
    }

private:
    template <typename in_type, typename out_type>
    void one_hot(Blob::Ptr input, Blob::Ptr output) {
        const auto *src_data = input->cbuffer().as<const in_type *>();
        auto *dst_data = output->buffer().as<out_type *>();
        std::size_t prefix_size = 1;
        auto input_dims = input->getTensorDesc().getDims();

        std::size_t actual_axis = (axis == -1) ? src_dims.size() : axis;
        for (size_t i = 0; i < actual_axis; ++i)
            prefix_size *= input_dims[i];

        std::size_t suffix_size = input->size() / prefix_size;

        // fill the output with off_value
        std::size_t dst_size = prefix_size * depth * suffix_size;
        std::fill(dst_data, dst_data + dst_size, off_value);

        // set on_value at needed locations
        parallel_for(prefix_size, [&](std::size_t prefix_idx) {
            for (std::size_t suffix_idx = 0; suffix_idx < suffix_size; ++suffix_idx) {
                auto src_index = prefix_idx * suffix_size + suffix_idx;
                auto v = static_cast<std::size_t>(src_data[src_index]);
                if (v < depth) {
                    std::size_t dst_offset = prefix_idx * depth * suffix_size + v * suffix_size + suffix_idx;
                    dst_data[dst_offset] = on_value;
                }
            }
        });
    }

    struct OneHotContext {
        OneHotImpl* nodePtr;
        Blob::Ptr input;
        Blob::Ptr output;
        bool executed;
    };

    template<typename T>
    struct OneHotExecute {
        using src_t = typename std::tuple_element<0, T>::type;
        using dst_t = typename std::tuple_element<1, T>::type;

        void operator()(OneHotContext & ctx) {
            ctx.nodePtr->one_hot<src_t, dst_t>(ctx.input, ctx.output);
            ctx.executed = true;
        }
    };

    uint32_t depth;
    float on_value = 1.f;
    float off_value = 0.f;
    int32_t axis = -1;
    SizeVector src_dims;
    SizeVector dst_dims;

    Precision input_precision;
    Precision output_precision;
};

REG_FACTORY_FOR(OneHotImpl, OneHot);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
