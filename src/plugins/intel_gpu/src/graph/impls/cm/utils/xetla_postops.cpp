// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "xetla_postops.hpp"

#include "graph/common_utils/jitter.hpp"
#include "xetla_helpers.hpp"

namespace ov::intel_gpu::cm {

std::vector<std::tuple<std::string, std::string>> XeTLAPostOPs::get_definitions() {
    std::string post_op_kernel_args = "";
    std::string post_op_args = "";
    std::string post_op_args_pass = "";
    std::string post_op_definitions = "";
    std::string post_op_list = "";
    std::string post_op_shape_definitions = "";
    std::string post_op_epilogue_init_args = "";

    bool first_epilogue_arg = true;
    bool first_post_op_list = true;

    for (const auto& post_op : postops) {
        auto kernel_arg_definition = post_op->get_kernel_arg_definition();
        if (!kernel_arg_definition.empty()) {
            post_op_kernel_args += ", " + kernel_arg_definition;
        }

        auto arg_definition = post_op->get_arg_definition();
        if (!arg_definition.empty()) {
            post_op_args += ", " + arg_definition;
        }

        auto arg_definition_pass = post_op->get_arg_name();
        if (!arg_definition_pass.empty()) {
            post_op_args_pass += ", " + arg_definition_pass;
        }

        post_op_definitions += post_op->get_definition();

        if (first_post_op_list) {
            post_op_list += post_op->get_definition_name();
            first_post_op_list = false;
        } else {
            post_op_list += ", " + post_op->get_definition_name();
        }

        post_op_shape_definitions += post_op->get_shape_definition();

        if (first_epilogue_arg) {
            post_op_epilogue_init_args += post_op->get_epilogue_init();
            first_epilogue_arg = false;
        } else {
            post_op_epilogue_init_args += ", " + post_op->get_epilogue_init();
        }
    }
    std::vector<std::tuple<std::string, std::string>> definitions;
    definitions.push_back({"XETLA_POST_OP_KERNEL_ARGS", post_op_kernel_args});
    definitions.push_back({"XETLA_POST_OP_ARGS", post_op_args});
    definitions.push_back({"XETLA_POST_OP_ARGS_PASS", post_op_args_pass});
    definitions.push_back({"XETLA_POST_OP_DEFINITIONS", post_op_definitions});
    definitions.push_back({"XETLA_POST_OP_LIST", post_op_list});
    definitions.push_back({"XETLA_POST_OP_SHAPE_DEFINITIONS", post_op_shape_definitions});
    definitions.push_back({"XETLA_POST_OP_EPILOGUE_INIT_ARGS", post_op_epilogue_init_args});

    return definitions;
}

size_t XeTLAPostOPs::add_post_ops(const RuntimeParams& params, size_t post_op_arg_index) {
    for (const auto& postop : params.fused_desc) {
        const bool is_eltwise = cldnn::fused_ops_are_one_of<cldnn::eltwise>({postop});
        const bool is_activation = cldnn::fused_ops_are_one_of<cldnn::activation>({postop});
        if (is_eltwise) {
            auto eltwise = std::static_pointer_cast<const cldnn::eltwise>(postop.desc);
            auto eltwise_layout = params.input_layouts[post_op_arg_index++];
            auto eltwise_dtype = ov_to_xetla_dtype(eltwise_layout.data_type);

            bool broadcast = false;
            bool is_M_dynamic = eltwise_layout.get_partial_shape()[0].is_dynamic() || eltwise_layout.get_partial_shape()[1].is_dynamic();
            if (!is_M_dynamic) {
                const auto eltwise_M = extract_channel(ChannelName::BATCH, eltwise_layout) * extract_channel(ChannelName::FEATURE, eltwise_layout);
                broadcast = eltwise_M == 1;
            }
            assert(eltwise->broadcast_spec.m_axis == 0);

            if (broadcast) {
                if (eltwise->mode == cldnn::eltwise_mode::sum) {
                    postops.push_back(std::make_unique<ShiftChannels>(post_op_index++, eltwise_dtype));
                } else if (eltwise->mode == cldnn::eltwise_mode::prod) {
                    postops.push_back(std::make_unique<ScaleChannels>(post_op_index++, eltwise_dtype));
                }
            } else {
                const auto eltwise_op = get_xetla_eltwise_op(eltwise->mode);
                assert(eltwise_op != Eltwise::EltwiseOp::none);
                postops.push_back(std::make_unique<Eltwise>(post_op_index++, eltwise_dtype, eltwise_op));
            }
        } else if (is_activation) {
            const auto activation = std::static_pointer_cast<const cldnn::activation>(postop.desc);
            const auto activation_dtype = ov_to_xetla_dtype(ov::element::Type_t::f32);
            const auto activation_op = get_xetla_activation_op(activation->activation_function);

            assert(activation_op != Activation::ActivationOp::none);
            postops.push_back(std::make_unique<Activation>(post_op_index++, activation_dtype, activation_op));
        }
    }
    return post_op_arg_index;
}

}  // namespace ov::intel_gpu::cm
