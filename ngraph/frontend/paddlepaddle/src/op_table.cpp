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
#include "op/fill_constant_batch_size_like.hpp"
#include "op/fill_constant.hpp"
#include "op/flatten_contiguous_range.hpp"
#include "op/interp.hpp"
#include "op/leakyrelu.hpp"
#include "op/matmul.hpp"
#include "op/mul.hpp"
#include "op/pad3d.hpp"
#include "op/pool2d.hpp"
#include "op/relu.hpp"
#include "op/reshape2.hpp"
#include "op/scale.hpp"
#include "op/slice.hpp"
#include "op/softmax.hpp"
#include "op/split.hpp"
#include "op/squeeze.hpp"
#include "op/unsqueeze.hpp"
#include "op/yolo_box.hpp"


#include "op_table.hpp"


namespace ngraph {
    namespace frontend {
        namespace pdpd {

            std::map<std::string, CreatorFunction> get_supported_ops() {
                return {
                        {"arg_max", op::argmax},
                        {"assign_value",                  op::assign_value},
                        {"batch_norm",                    op::batch_norm},
                        {"bilinear_interp_v2",            op::bilinear_interp_v2},
                        {"bilinear_interp",               op::bilinear_interp_v2},
                        {"cast",                          op::cast},
                        {"clip",                          op::clip},
                        {"concat",                        op::concat},
                        {"conv2d",                        op::conv2d},
                        {"elementwise_add",               op::elementwise_add},
                        {"elementwise_div",               op::elementwise_div},
                        {"elementwise_max",               op::elementwise_max},
                        {"elementwise_min",               op::elementwise_min},
                        {"elementwise_mul",               op::elementwise_mul},
                        {"elementwise_pow",               op::elementwise_pow},
                        {"elementwise_sub",               op::elementwise_sub},
                        {"fill_constant_batch_size_like", op::fill_constant_batch_size_like},
                        {"fill_constant",                 op::fill_constant},
                        {"flatten_contiguous_range",      op::flatten_contiguous_range},
                        {"leaky_relu",                    op::leaky_relu},
                        {"matmul",                        op::matmul},
                        {"mul",                           op::mul},
                        {"nearest_interp_v2",             op::nearest_interp_v2},
                        {"nearest_interp",                op::nearest_interp_v2},
                        {"pad3d",                         op::pad3d},
                        {"pool2d",                        op::pool2d},
                        {"relu",                          op::relu},
                        {"reshape2",                      op::reshape2},
                        {"scale",                         op::scale},
                        {"slice",                         op::slice},
                        {"softmax",                       op::softmax},
                        {"split",                         op::split},
                        {"squeeze2",                      op::squeeze},
                        {"unsqueeze2",                    op::unsqueeze},
                        {"yolo_box",                      op::yolo_box}
                };
            };

        }
    }
}
