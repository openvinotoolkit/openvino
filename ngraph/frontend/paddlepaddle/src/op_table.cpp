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
#include "op/elementwise_ops.hpp"
#include "op/mul.hpp"
#include "op/pad3d.hpp"
#include "op/pool2d.hpp"
#include "op/pow.hpp"
#include "op/range.hpp"
#include "op/relu.hpp"
#include "op/relu6.hpp"
#include "op/reshape2.hpp"
#include "op/scale.hpp"
#include "op/shape.hpp"
#include "op/sigmoid.hpp"
#include "op/slice.hpp"
#include "op/softmax.hpp"
#include "op/split.hpp"
#include "op/squeeze.hpp"
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
                        {"cast", op::cast},
                        {"clip", op::clip},
                        {"concat", op::concat},
                        {"conv2d", op::conv2d},
                        {"elementwise_add", op::elementwise_add},
                        {"elementwise_div", op::elementwise_div},
                        {"elementwise_max", op::elementwise_max},
                        {"elementwise_min", op::elementwise_min},
                        {"elementwise_mul", op::elementwise_mul},
                        {"elementwise_pow", op::elementwise_pow},
                        {"elementwise_sub", op::elementwise_sub},
                        {"max_pool2d_with_index", op::pool2d},
                        {"mul", op::mul},
                        {"pad3d", op::pad3d},
                        {"pow", op::pow},
                        {"pool2d", op::pool2d},
                        {"range", op::range},
                        {"relu", op::relu},
                        {"relu6", op::relu6},
                        {"reshape2", op::reshape2},
                        {"scale", op::scale},
                        {"shape", op::shape},
                        {"slice", op::slice},
                        {"softmax", op::softmax},
                        {"sigmoid", op::sigmoid},
                        {"split", op::split},
                        {"squeeze2", op::squeeze},
                        {"sync_batch_norm", op::batch_norm},
                        {"unsqueeze2", op::unsqueeze},
                        {"yolo_box", op::yolo_box}};
            };

        } // namespace pdpd
    }     // namespace frontend
} // namespace ngraph
