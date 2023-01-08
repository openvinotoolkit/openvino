// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils/conv_factory.hpp"

#include "default_opset.hpp"
#include "exceptions.hpp"
#include "ngraph/builder/reshape.hpp"
#include "ngraph/op/group_conv.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "onnx_import/core/null_node.hpp"
#include "utils/conv_factory.hpp"
#include "utils/convpool.hpp"
#include "utils/reshape.hpp"

namespace ngraph {
namespace onnx_import {
namespace conv_factory {
std::shared_ptr<ov::op::Op> make_ng_convolution(const Output<ngraph::Node>& data,
                                                const Output<ngraph::Node>& filters,
                                                const ngraph::Strides& strides,
                                                const ngraph::Strides& dilations,
                                                const ngraph::CoordinateDiff& padding_below,
                                                const ngraph::CoordinateDiff& padding_above,
                                                int64_t groups,
                                                const ngraph::op::PadType& auto_pad) {
    if (groups > 1) {
        const auto reshaped_filters = convpool::get_reshaped_filters(filters, groups);

        return std::make_shared<default_opset::GroupConvolution>(data,
                                                                 reshaped_filters,
                                                                 strides,
                                                                 padding_below,
                                                                 padding_above,
                                                                 dilations,
                                                                 auto_pad);
    } else {
        return std::make_shared<default_opset::Convolution>(data,
                                                            filters,
                                                            strides,
                                                            padding_below,
                                                            padding_above,
                                                            dilations,
                                                            auto_pad);
    }
}
}  // namespace conv_factory
}  // namespace onnx_import
}  // namespace ngraph
