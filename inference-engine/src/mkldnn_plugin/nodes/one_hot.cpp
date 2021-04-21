// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "base.hpp"
#include "ie_parallel.hpp"
#include "common/tensor_desc_creator.h"
#include "common/cpu_memcpy.h"
#include "utils/bfloat16.hpp"
#include <mkldnn_selective_build.h>
#include <ngraph/opsets/opset1.hpp>

#include <vector>

using namespace MKLDNNPlugin;

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class OneHotImpl: public ExtLayerBase {
    typedef PrecisionTrait<Precision::I32>::value_type in_type;

    bool isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept {
        try {
            const auto oneHot = std::dynamic_pointer_cast<const ngraph::opset1::OneHot>(op);
            if (!oneHot) {
                errorMessage = "Only opset1 OneHot operation is supported";
                return false;
            }
            if (std::dynamic_pointer_cast<const ngraph::opset1::Constant>(oneHot->get_input_node_shared_ptr(DEPTH_ID)) == nullptr) {
                errorMessage = "Only const 'depth' input is supported";
                return false;
            }
            if (std::dynamic_pointer_cast<const ngraph::opset1::Constant>(oneHot->get_input_node_shared_ptr(ON_VALUE_ID)) == nullptr) {
                errorMessage = "Only const 'on_value' input is supported";
                return false;
            }
            if (std::dynamic_pointer_cast<const ngraph::opset1::Constant>(oneHot->get_input_node_shared_ptr(OFF_VALUEAXES_ID)) == nullptr) {
                errorMessage = "Only const 'off_value' input is supported";
                return false;
            }
        } catch (...) {
            return false;
        }
        return true;
    }

public:
    explicit OneHotImpl(const std::shared_ptr<ngraph::Node>& op) {
        try {
            std::string errorMessage;
            if (!isSupportedOperation(op, errorMessage)) {
                IE_THROW(NotImplemented) << errorMessage;
            }

            errorPrefix = "OneHot layer with name '" + op->get_friendly_name() + "'";
            const auto oneHot = std::dynamic_pointer_cast<const ngraph::opset1::OneHot>(op);
            const auto depthNode = std::dynamic_pointer_cast<const ngraph::opset1::Constant>(oneHot->get_input_node_shared_ptr(DEPTH_ID));
            const auto onValueNode = std::dynamic_pointer_cast<const ngraph::opset1::Constant>(oneHot->get_input_node_shared_ptr(ON_VALUE_ID));
            const auto offValueNode = std::dynamic_pointer_cast<const ngraph::opset1::Constant>(oneHot->get_input_node_shared_ptr(OFF_VALUEAXES_ID));
            depth = depthNode->cast_vector<uint32_t>()[0];
            axis = oneHot->get_axis();
            src_dims = oneHot->get_input_shape(INDICES_ID);
            if (ngraph::is_scalar(src_dims)) {
                src_dims = SizeVector{1};
            }
            dst_dims = oneHot->get_output_shape(0);
            if (ngraph::is_scalar(dst_dims)) {
                dst_dims = SizeVector{1};
            }

            int output_dims_size = dst_dims.size();
            if (axis < 0) {
                axis += output_dims_size;
            }
            if (axis < 0 || axis >= output_dims_size) {
                IE_THROW() << errorPrefix << " has unsupported 'axis' attribute: " << oneHot->get_axis();
            }

            if (!( ((1 + src_dims.size()) == dst_dims.size()) ||
                   (src_dims.size() == 1 && dst_dims.size() == 1 && dst_dims[0] == depth && src_dims[0] == 1)))
                IE_THROW() << errorPrefix << " has incorrect number of input/output dimensions!";

            // check a precision of the input tensor
            auto input_precision = details::convertPrecision(oneHot->get_input_element_type(INDICES_ID));
            if (input_precision != Precision::I32) {
                IE_THROW() << errorPrefix << " has incorrect input precision for the input. Only I32 is supported!";
            }
            output_precision = details::convertPrecision(oneHot->get_output_element_type(0));
            if (Precision::BF16 == output_precision) {
                MKLDNNPlugin::bfloat16_t bf16_on_value  = onValueNode->cast_vector<float>()[0];
                MKLDNNPlugin::bfloat16_t bf16_off_value = offValueNode->cast_vector<float>()[0];
                cpu_memcpy(&on_value, &bf16_on_value, sizeof(MKLDNNPlugin::bfloat16_t));
                cpu_memcpy(&off_value, &bf16_off_value, sizeof(MKLDNNPlugin::bfloat16_t));
            } else if (output_precision.is_float()) {
                float float_on_value  = onValueNode->cast_vector<float>()[0];
                float float_off_value = offValueNode->cast_vector<float>()[0];
                cpu_memcpy(&on_value, &float_on_value, sizeof(float));
                cpu_memcpy(&off_value, &float_off_value, sizeof(float));
            } else {
                on_value = onValueNode->cast_vector<int>()[0];
                off_value = offValueNode->cast_vector<int>()[0];
            }

            addConfig(op, {{TensorDescCreatorTypes::ncsp, input_precision},
                           {TensorDescCreatorTypes::ncsp, input_precision},
                           {TensorDescCreatorTypes::ncsp, output_precision},
                           {TensorDescCreatorTypes::ncsp, output_precision}},
                          {{TensorDescCreatorTypes::ncsp, output_precision}});
        } catch (InferenceEngine::Exception& ex) {
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

    std::string errorPrefix;

    static const size_t INDICES_ID = 0;
    static const size_t DEPTH_ID = 1;
    static const size_t ON_VALUE_ID = 2;
    static const size_t OFF_VALUEAXES_ID = 3;
};

REG_FACTORY_FOR(OneHotImpl, OneHot);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
