// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "op/argmax.hpp"
#include "op/assign_value.hpp"
#include "op/batch_norm.hpp"
#include "op/cast.hpp"
#include "op/clip.hpp"
#include "op/concat.hpp"
#include "op/conv2d.hpp"
#include "op/conv2d_transpose.hpp"
#include "op/dropout.hpp"
#include "op/elementwise_ops.hpp"
#include "op/equal.hpp"
#include "op/expand_v2.hpp"
#include "op/fill_constant.hpp"
#include "op/fill_constant_batch_size_like.hpp"
#include "op/flatten_contiguous_range.hpp"
#include "op/greater_equal.hpp"
#include "op/hard_sigmoid.hpp"
#include "op/hard_swish.hpp"
#include "op/interp.hpp"
#include "op/leakyrelu.hpp"
#include "op/log.hpp"
#include "op/logical_not.hpp"
#include "op/matmul.hpp"
#include "op/relu.hpp"
#include "op/rnn.hpp"
#include "op/scale.hpp"
#include "op/split.hpp"
#include "op/transpose2.hpp"

#include "op_table.hpp"

namespace ngraph
{
    namespace frontend
    {
        namespace pdpd
        {
            std::map<std::string, CreatorFunction> get_supported_ops()
            {
                return {
                    {"arg_max", op::argmax},
                    {"assign_value", op::assign_value},
                    {"batch_norm", op::batch_norm},
                    {"bilinear_interp_v2", op::bilinear_interp_v2},
                    {"bilinear_interp", op::bilinear_interp_v2},
                    {"bmm", op::matmul},
                    {"cast", op::cast},
                    {"clip", op::clip},
                    {"concat", op::concat},
                    {"conv2d", op::conv2d},
                    {"conv2d_transpose", op::conv2d_transpose},
                    {"depthwise_conv2d", op::conv2d},
                    {"depthwise_conv2d_transpose", op::conv2d_transpose},
                    {"dropout", op::dropout},
                    {"elementwise_add", op::elementwise_add},
                    {"elementwise_div", op::elementwise_div},
                    {"elementwise_max", op::elementwise_max},
                    {"elementwise_min", op::elementwise_min},
                    {"elementwise_mul", op::elementwise_mul},
                    {"elementwise_pow", op::elementwise_pow},
                    {"elementwise_sub", op::elementwise_sub},
                    {"equal", op::equal},
                    {"expand_v2", op::expand_v2},
                    {"fill_constant_batch_size_like", op::fill_constant_batch_size_like},
                    {"fill_constant", op::fill_constant},
                    {"flatten_contiguous_range", op::flatten_contiguous_range},
                    {"greater_equal", op::greater_equal},
                    {"hard_sigmoid", op::hard_sigmoid},
                    {"hard_swish", op::hard_swish},
                    {"leaky_relu", op::leaky_relu},
                    {"log", op::log},
                    {"logical_not", op::logical_not},
                    {"matmul", op::matmul},
                    {"nearest_interp_v2", op::nearest_interp_v2},
                    {"nearest_interp", op::nearest_interp_v2},
                    {"rnn", op::rnn},
                    {"relu", op::relu},
                    {"scale", op::scale},
                    {"split", op::split},
                    {"transpose2", op::transpose2},
                };
            };

        } // namespace pdpd
    }     // namespace frontend
} // namespace ngraph
