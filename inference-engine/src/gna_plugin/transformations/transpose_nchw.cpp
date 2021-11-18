// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/cc/ngraph/itt.hpp>

#include "transformations/transpose_nchw.hpp"

#include "transformations/utils/transformation_helper.hpp"

#include <ngraph_ops/convolution_ie.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <transformations/utils/utils.hpp>
#include <ops/gna_convolution.hpp>

#include <vector>

using namespace GNAPluginNS;

NGRAPH_RTTI_DEFINITION(TransposeNCHW, "TransposeNCHW", 0);

using Node = std::shared_ptr<ngraph::Node>;

namespace
{

/* transpose orders
   before convolution convert NCHW -> NHWC
    3D: NCX {0, 1, 2} -> NXC {0, 2, 1}
    4D: NCHW {0, 1, 2, 3} -> NHWC {0, 2, 3, 1}
    5D: NCZYX {0, 1, 2, 3, 4} -> NZYXC {0, 2, 3, 4, 1}
   
   after convolution convert NHWC -> NCHW
   3D: NXC {0, 1, 2} -> NCX {0, 2, 1}
   4D: NHWC {0, 1, 2, 3} -> NCHW {0, 3, 1, 2}
   5D: NZYXC {0, 1, 2, 3} -> NCZYX {0, 4, 1, 2, 3}

   so just
   1) temp = A[N - 1]
   2) move A[j] -> A[j + 1] for 1 <= j <= N - 2
   3) A[1] = temp
*/

ngraph::Shape GenerateTransposeOrderNCHW2NHWC(size_t shape_size)
{
    ngraph::Shape shape(shape_size);
    std::iota(shape.begin(), shape.end(), 0);

    for (int i = 1; i < shape.size() - 1; ++i)
        shape[i] = shape[i + 1];

    *(shape.end() - 1) = 1;

    return shape;
}

ngraph::Shape GenerateTransposeOrderNHWC2NCHW(size_t shape_size)
{
    ngraph::Shape shape(shape_size);
    std::iota(shape.begin(), shape.end(), 0);

    const size_t channels_position = *(shape.end() - 1);

    for (int i = shape.size() - 1; i > 0; --i)
        shape[i] = shape[i - 1];

    shape[1] = channels_position;

    return shape;
}

bool DoTransformation(Node convolution)
{
    auto convolution_node = std::dynamic_pointer_cast<ngraph::opset8::Convolution>(convolution);
    auto convolution_input_data_node = convolution_node->input_value(0);
    auto convolution_input_const_node = convolution_node->input_value(1);
    const ngraph::Shape convolution_input_shape = convolution_node->get_input_shape(0);
    const ngraph::Shape convolution_out_shape = convolution_node->get_output_shape(0);

    if (convolution_input_shape.size() < 3 || convolution_input_shape.size() > 5)
        return false;

    const ngraph::Shape transpose_before_order = GenerateTransposeOrderNCHW2NHWC(convolution_input_shape.size());

    auto transpose_const = ngraph::opset8::Constant::create(ngraph::element::i64,
                                                            ngraph::Shape{transpose_before_order.size()},
                                                            transpose_before_order);

    auto transpose_before = std::make_shared<ngraph::opset8::Transpose>(convolution_input_data_node,
                                                                        transpose_const);

    auto transpose_conv_constant = std::make_shared<ngraph::opset8::Transpose>(convolution_input_const_node,
                                                                               transpose_const);
    auto conv_new = std::make_shared<GNAPluginNS::GNAConvolution>(transpose_before,
                                                                   transpose_conv_constant,
                                                                   convolution_node->get_strides(),
                                                                   convolution_node->get_pads_begin(),
                                                                   convolution_node->get_pads_end(),
                                                                   convolution_node->get_dilations(),
                                                                   convolution_node->get_auto_pad());

    const ngraph::Shape transpose_after_order = GenerateTransposeOrderNHWC2NCHW(conv_new->get_output_shape(0).size());

    auto transpose_after = std::make_shared<ngraph::opset8::Transpose>(conv_new,
                                                                       ngraph::opset8::Constant::create(ngraph::element::i64,
                                                                       ngraph::Shape{transpose_after_order.size()},
                                                                       transpose_after_order));    

    ngraph::copy_runtime_info(convolution_node, transpose_before);
    transpose_before->set_friendly_name(convolution_node->get_friendly_name() + "/gna_conv_transpose_before");

    ngraph::copy_runtime_info(convolution_node, conv_new);
    conv_new->set_friendly_name(convolution_node->get_friendly_name() + "/gna_convolution");

    ngraph::copy_runtime_info(convolution_node, transpose_after);
    transpose_after->set_friendly_name(convolution_node->get_friendly_name() + "/gna_conv_transpose_after");

    convolution->output(0).replace(transpose_after->output(0));
    return true;
}

} // namespace

TransposeNCHW::TransposeNCHW() {
    MATCHER_SCOPE(TransposeNCHW);

    auto convolution = ngraph::pattern::wrap_type<ngraph::opset8::Convolution>();

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto convolution_node = std::dynamic_pointer_cast<ngraph::opset8::Convolution>(m.get_match_root());
        if (!convolution_node) {
            return false;
        }

        return DoTransformation(convolution_node);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(convolution, matcher_name);
    this->register_matcher(m, callback);
}
