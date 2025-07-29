// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <openvino/openvino.hpp>
#include <openvino/frontend/node_context.hpp>
#include <openvino/frontend/paddle/node_context.hpp>
//! [add_extension_header]
//#include <openvino/core/op_extension.hpp>
//! [add_extension_header]
//! [add_frontend_extension_header]
#include <openvino/frontend/extension.hpp>
//! [add_frontend_extension_header]

//! [frontend_extension_Identity_header]
#include <openvino/frontend/extension.hpp>
//! [frontend_extension_Identity_header]

//! [frontend_extension_ThresholdedReLU_header]
#include <openvino/opsets/opset11.hpp>
//! [frontend_extension_ThresholdedReLU_header]

//! [frontend_extension_framework_map_macro_headers]
#include <openvino/frontend/extension/op.hpp>
#include <openvino/frontend/onnx/extension/op.hpp>
#include <openvino/frontend/tensorflow/extension/op.hpp>
#include <openvino/frontend/paddle/extension/op.hpp>
//! [frontend_extension_framework_map_macro_headers]

// #include <identity.hpp>


class RotRPEAttentionWeightWithIndexComputation : public ov::op::Op {
public:
    OPENVINO_OP("RotRPEAttentionWeightWithIndexComputation");

    RotRPEAttentionWeightWithIndexComputation() = default;
    
    RotRPEAttentionWeightWithIndexComputation(const ov::OutputVector& args) 
        : Op(args) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        // Validate input count
        NODE_VALIDATION_CHECK(this, get_input_size() == 4,
            "Expected 4 inputs. Got: ", get_input_size());

        // Validate input shapes
        const auto& query_shape = get_input_partial_shape(0);
        const auto& key_shape = get_input_partial_shape(1);
        const auto& index_shape = get_input_partial_shape(2);
        const auto& pos_shape = get_input_partial_shape(3);

        NODE_VALIDATION_CHECK(this, query_shape.rank().is_static() && query_shape.rank() == 3,
            "Query features must be 3-dimensional");
        NODE_VALIDATION_CHECK(this, key_shape.rank().is_static() && key_shape.rank() == 3,
            "Key features must be 3-dimensional");
        NODE_VALIDATION_CHECK(this, index_shape.rank().is_static() && index_shape.rank() == 3,
            "Key attention index must be 3-dimensional");
        NODE_VALIDATION_CHECK(this, pos_shape.rank().is_static() && pos_shape.rank() == 4,
            "QK relative position must be 4-dimensional");

        // // Infer output shape [batch_size, nhead, key_index_size]
        // auto batch_size = pos_shape[0];
        // auto nhead = query_shape[1] / batch_size;
        // auto key_index_size = index_shape[2];
        
        // set_output_type(0, get_input_element_type(0), 
        //                PartialShape({batch_size, nhead, key_index_size}));

        // 严格保持原始ORT的输出形状逻辑
        std::vector<ov::Dimension> output_dims = {
            query_shape[1],  // dimensions_0[1] (nhead * bs)
            query_shape[0],   // dimensions_0[0] (total_query_num)
            index_shape[2]    // dimensions_2[2] (key_index_size)
        };

        set_output_type(0, get_input_element_type(0), ov::PartialShape(output_dims));
    }

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override {
        return std::make_shared<RotRPEAttentionWeightWithIndexComputation>(new_args);
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override {
        // Extract input data pointers
        const float* query_features = inputs[0].data<const float>();
        const float* key_features = inputs[1].data<const float>();
        const int32_t* key_attn_index = inputs[2].data<const int32_t>();
        const float* qk_relative_pos = inputs[3].data<const float>();
        float* output = outputs[0].data<float>();

        // Get shapes
        const auto& query_shape = inputs[0].get_shape();
        const auto& key_shape = inputs[1].get_shape();
        const auto& index_shape = inputs[2].get_shape();
        const auto& pos_shape = inputs[3].get_shape();

        // Extract dimensions
        const int32_t bs = pos_shape[0];
        const int32_t total_query_num = query_shape[0];
        const int32_t total_key_num = key_shape[0];
        const int32_t key_index_size = index_shape[2];
        const int32_t nhead = query_shape[1] / bs;
        const int32_t hdim = query_shape[2];

        // Allocate frequency buffer
        std::vector<float> mFreq(nhead * hdim, 0);
        get_rel_pos_freq_kernel_trt(mFreq.data(), nhead * hdim, pos_shape[3]);

        // Call computation kernel
        attention_weight_computation_forward_rot_ort_local(
            bs, total_query_num, total_key_num, key_index_size, nhead, hdim,
            query_features, key_features, key_attn_index,
            qk_relative_pos, mFreq.data(), output
        );

        return true;
    }

    bool has_evaluate() const override {
        return true;
    }

private:
    // Placeholder for the actual implementation functions
    void get_rel_pos_freq_kernel_trt(float* freq_buffer, int size, int64_t dim3) const {
        // TODO: Implement frequency calculation
    }

    void attention_weight_computation_forward_rot_ort_local(
        int32_t bs, int32_t total_query_num, int32_t total_key_num, 
        int32_t key_index_size, int32_t nhead, int32_t hdim,
        const float* query_features, const float* key_features, 
        const int32_t* key_attn_index, const float* qk_relative_pos,
        const float* mFreq, float* output) const {
        // TODO: Implement attention weight computation
    }
};

// 注册扩展
OPENVINO_CREATE_EXTENSIONS(
    std::vector<std::shared_ptr<ov::Extension>>({
        std::make_shared<ov::OpExtension<RotRPEAttentionWeightWithIndexComputation>>()
    });
)