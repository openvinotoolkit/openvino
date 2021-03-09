// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "base.hpp"
#include "ie_parallel.hpp"
#include "common/tensor_desc_creator.h"
#include "common/cpu_memcpy.h"
#include "utils/bfloat16.hpp"
#include <mkldnn_selective_build.h>

#include <vector>

using namespace MKLDNNPlugin;

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class OneHotImpl: public ExtLayerBase {
    typedef PrecisionTrait<Precision::I32>::value_type in_type;

public:
    explicit OneHotImpl(const CNNLayer* layer) {
        try {
            depth     = layer->GetParamAsUInt("depth");
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
            auto input_precision = layer->insData[0].lock()->getTensorDesc().getPrecision();
            if (input_precision != Precision::I32) {
                THROW_IE_EXCEPTION << layer->name << " Incorrect input precision for the input. Only I32 is supported!";
            }
            output_precision = layer->outData[0]->getTensorDesc().getPrecision();
            if (Precision::BF16 == output_precision) {
                MKLDNNPlugin::bfloat16_t bf16_on_value  = layer->GetParamAsFloat("on_value", 1.0f);
                MKLDNNPlugin::bfloat16_t bf16_off_value = layer->GetParamAsFloat("off_value", 0.0f);
                cpu_memcpy(&on_value, &bf16_on_value, sizeof(MKLDNNPlugin::bfloat16_t));
                cpu_memcpy(&off_value, &bf16_off_value, sizeof(MKLDNNPlugin::bfloat16_t));
            } else if (output_precision.is_float()) {
                float float_on_value  = layer->GetParamAsFloat("on_value", 1.0f);
                float float_off_value = layer->GetParamAsFloat("off_value", 0.0f);
                cpu_memcpy(&on_value, &float_on_value, sizeof(float));
                cpu_memcpy(&off_value, &float_off_value, sizeof(float));
            } else {
                on_value = layer->GetParamAsInt("on_value", 1);
                off_value = layer->GetParamAsInt("off_value", 0);
            }

            LayerConfig config;
            DataConfig dataConfig;
            config.dynBatchSupport = false;

            auto& creators = MKLDNNPlugin::TensorDescCreator::getCommonCreators();

            dataConfig.desc = creators.at(MKLDNNPlugin::TensorDescCreatorTypes::ncsp)->createDesc(input_precision, src_dims);
            config.inConfs.push_back(dataConfig);

            dataConfig.desc = creators.at(MKLDNNPlugin::TensorDescCreatorTypes::ncsp)->createDesc(output_precision, dst_dims);
            config.outConfs.push_back(dataConfig);

            confs.push_back(config);
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs, ResponseDesc *resp) noexcept override {
        try {
            std::size_t prefix_size = 1;
            auto input_dims = inputs.front()->getTensorDesc().getDims();

            std::size_t actual_axis = (axis == -1) ? src_dims.size() : axis;
            for (size_t i = 0; i < actual_axis; ++i)
                prefix_size *= input_dims[i];

            std::size_t suffix_size = inputs.front()->size() / prefix_size;

            OneHotContext ctx = {this, inputs[0], outputs[0], prefix_size, suffix_size, false};
            OV_SWITCH(MKLDNNPlugin, OneHotExecute, ctx, output_precision.size(),
                      OV_CASE(sizeof(uint32_t), uint32_t),
                      OV_CASE(sizeof(uint16_t), uint16_t),
                      OV_CASE(sizeof(uint8_t), uint8_t))

            if (!ctx.executed) {
                snprintf(resp->msg, sizeof(resp->msg), "Unsupported output data type %s.", output_precision.name());
                return GENERAL_ERROR;
            }
        }
        catch (const std::exception& excp) {
            snprintf(resp->msg, sizeof(resp->msg), "%s", excp.what());
            return GENERAL_ERROR;
        }
        catch(...) {
            return GENERAL_ERROR;
        }
        return OK;
    }

private:
    template<typename out_type>
    void one_hot(const Blob::Ptr& input, const Blob::Ptr& output, size_t prefix_size, size_t suffix_size) {
        const auto *src_data = input->cbuffer().as<const in_type *>();
        auto *dst_data = output->buffer().as<out_type *>();

        // fill the output with off_value
        std::size_t dst_size = prefix_size * depth * suffix_size;
        std::fill(dst_data, dst_data + dst_size, static_cast<out_type>(off_value));

        // set on_value at needed locations
        auto on_val = static_cast<out_type>(on_value);
        parallel_for(prefix_size, [&](std::size_t prefix_idx) {
            const in_type* src_dataPtr = &src_data[prefix_idx * suffix_size];
            out_type* dst_dataPtr = &dst_data[prefix_idx * depth * suffix_size];
            for (std::size_t suffix_idx = 0; suffix_idx < suffix_size; ++suffix_idx, ++src_dataPtr, ++dst_dataPtr) {
                auto v = static_cast<std::size_t>(*src_dataPtr);
                if (v < depth) {
                    dst_dataPtr[v * suffix_size] = on_val;
                }
            }
        });
    }

    struct OneHotContext {
        OneHotImpl* nodePtr;
        Blob::Ptr input;
        Blob::Ptr output;
        size_t prefix_size;
        size_t suffix_size;
        bool executed;
    };

    template<typename dst_t>
    struct OneHotExecute {
        void operator()(OneHotContext & ctx) {
            ctx.nodePtr->one_hot<dst_t>(ctx.input, ctx.output, ctx.prefix_size, ctx.suffix_size);
            ctx.executed = true;
        }
    };

    uint32_t depth;
    uint32_t on_value;
    uint32_t off_value;
    int32_t axis = -1;
    SizeVector src_dims;
    SizeVector dst_dims;

    Precision output_precision;
};

REG_FACTORY_FOR(OneHotImpl, OneHot);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
