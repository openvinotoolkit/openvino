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
#include "op/mul.hpp"
#include "op/pad3d.hpp"
#include "op/pool2d.hpp"
#include "op/pow.hpp"
#include "op/range.hpp"
#include "op/relu.hpp"
#include "op/relu6.hpp"
#include "op/reshape2.hpp"
#include "op/rnn.hpp"
#include "op/scale.hpp"
#include "op/shape.hpp"
#include "op/sigmoid.hpp"
#include "op/slice.hpp"
#include "op/softmax.hpp"
#include "op/split.hpp"
#include "op/squeeze.hpp"
#include "op/transpose2.hpp"
#include "op/unsqueeze.hpp"
#include "op/yolo_box.hpp"

#include "op_table.hpp"

namespace ngraph
{
    namespace frontend
    {
        namespace pdpd
        {
            std::map<std::string, CreatorFunction> get_supported_ops()
            {
                return {{"arg_max", op::argmax},
                        {"assign_value", op::assign_value},
                        {"batch_norm", op::batch_norm},
                        {"bilinear_interp_v2", op::bilinear_interp_v2},
                        {"bilinear_interp", op::bilinear_interp_v2},
                        {"bmm", op::matmul},
                        {"cast", op::cast},
                        {"clip", op::clip},
                        {"concat", op::concat},
                        {"conv2d", op::conv2d},
                        {"depthwise_conv2d", op::conv2d},
                        {"depthwise_conv2d_transpose", op::conv2d_transpose},
                        {"dropout", op::dropout},
                        {"conv2d_transpose", op::conv2d_transpose},
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
                        {"max_pool2d_with_index", op::pool2d},
                        {"mul", op::mul},
                        {"nearest_interp_v2", op::nearest_interp_v2},
                        {"nearest_interp", op::nearest_interp_v2},
                        {"pad3d", op::pad3d},
                        {"pow", op::pow},
                        {"pool2d", op::pool2d},
                        {"range", op::range},
                        {"relu", op::relu},
                        {"relu6", op::relu6},
                        {"reshape2", op::reshape2},
                        {"rnn", op::rnn},
                        {"scale", op::scale},
                        {"shape", op::shape},
                        {"slice", op::slice},
                        {"softmax", op::softmax},
                        {"sigmoid", op::sigmoid},
                        {"split", op::split},
                        {"squeeze2", op::squeeze},
                        {"transpose2", op::transpose2},
                        {"unsqueeze2", op::unsqueeze},
                        {"yolo_box", op::yolo_box}};
            };

        } // namespace pdpd
    }     // namespace frontend
} // namespace ngraph
