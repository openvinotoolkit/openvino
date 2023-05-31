// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/transpose_nchw.hpp"

#include <ngraph/pass/manager.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <openvino/cc/ngraph/itt.hpp>
#include <openvino/opsets/opset10.hpp>
#include <ops/gna_convolution.hpp>
#include <ops/gna_max_pool.hpp>
#include <transformations/utils/utils.hpp>
#include <vector>

#include "transformations/utils/transformation_helper.hpp"

using namespace ov;
using namespace ov::opset10;
using namespace ov::pass::pattern;
using namespace ov::intel_gna::pass;
using namespace ov::intel_gna::pass::helper;

NGRAPH_RTTI_DEFINITION(ov::intel_gna::pass::TransposeNCHW, "TransposeNCHW");
NGRAPH_RTTI_DEFINITION(ov::intel_gna::pass::SubstituteGNAConvolution, "SubstituteGNAConvolution");
NGRAPH_RTTI_DEFINITION(ov::intel_gna::pass::SubstituteGNAMaxPool, "SubstituteGNAMaxPool");

namespace {
ov::Shape MakeTransposeOrderNCHW2NHWC(size_t shape_size);
ov::Shape MakeTransposeOrderNHWC2NCHW(size_t shape_size);

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

ov::Shape MakeTransposeOrderNCHW2NHWC(size_t shape_size) {
    ov::Shape shape(shape_size);
    std::iota(shape.begin(), shape.end(), 0);

    for (int i = 1; i < shape.size() - 1; ++i)
        shape[i] = shape[i + 1];

    *(shape.end() - 1) = 1;

    return shape;
}

ov::Shape MakeTransposeOrderNHWC2NCHW(size_t shape_size) {
    ov::Shape shape(shape_size);
    std::iota(shape.begin(), shape.end(), 0);

    const size_t channels_position = *(shape.end() - 1);

    for (int i = shape.size() - 1; i > 0; --i)
        shape[i] = shape[i - 1];

    shape[1] = channels_position;

    return shape;
}

template <typename T>
bool HasParentNode(std::shared_ptr<ov::Node> node) {
    for (const auto& parent : node->input_values()) {
        if (dynamic_cast<const T*>(parent.get_node()))
            return true;
    }
    return false;
}

template <typename T>
bool HasChildNode(std::shared_ptr<ov::Node> node) {
    for (size_t output_idx = 0; output_idx < node->get_output_size(); ++output_idx) {
        for (auto& input : node->get_output_target_inputs(output_idx)) {
            if (dynamic_cast<const T*>(input.get_node()))
                return true;
        }
    }
    return false;
}

}  // namespace

namespace SubstituteGNAConvolutionNS {

bool DoTransformation(std::shared_ptr<ov::Node> convolution);

bool DoTransformation(std::shared_ptr<ov::Node> convolution) {
    auto convolution_node = std::dynamic_pointer_cast<Convolution>(convolution);
    auto convolution_input_data_node = convolution_node->input_value(0);
    auto convolution_input_const_node = convolution_node->input_value(1);
    const ov::Shape convolution_input_shape = convolution_node->get_input_shape(0);

    // TODO: check input_data_node is not Reshape since that pattern should be matched in another transformation

    if (convolution_input_shape.size() != 3 && convolution_input_shape.size() != 4) {
        std::cout << "TransposeNCHW: unsupported convolution size " << convolution_input_shape.size() << std::endl;
        return false;
    }

    const ov::Shape transpose_before_order = MakeTransposeOrderNCHW2NHWC(convolution_input_shape.size());

    auto transpose_const =
        Constant::create(element::i64, ov::Shape{transpose_before_order.size()}, transpose_before_order);

    auto transpose_before = std::make_shared<Transpose>(convolution_input_data_node, transpose_const);

    auto transpose_conv_constant = std::make_shared<Transpose>(convolution_input_const_node, transpose_const);
    auto conv_new = std::make_shared<ov::intel_gna::op::GNAConvolution>(transpose_before,
                                                                        transpose_conv_constant,
                                                                        convolution_node->get_strides(),
                                                                        convolution_node->get_pads_begin(),
                                                                        convolution_node->get_pads_end(),
                                                                        convolution_node->get_dilations(),
                                                                        convolution_node->get_auto_pad());

    const ov::Shape transpose_after_order = MakeTransposeOrderNHWC2NCHW(conv_new->get_output_shape(0).size());

    auto transpose_after = std::make_shared<Transpose>(
        conv_new,
        Constant::create(element::i64, ov::Shape{transpose_after_order.size()}, transpose_after_order));

    ov::copy_runtime_info(convolution_node,
                          {transpose_before, transpose_const, conv_new, transpose_after, transpose_conv_constant});

    ov::replace_output_update_name(convolution->output(0), transpose_after->output(0));

    return true;
}

}  // namespace SubstituteGNAConvolutionNS

namespace SubstituteGNAMaxPoolNS {

bool DoTransformation(std::shared_ptr<ov::Node> convolution);

bool DoTransformation(std::shared_ptr<ov::Node> max_pool) {
    auto max_pool_node = std::dynamic_pointer_cast<ov::op::v1::MaxPool>(max_pool);
    auto max_pool_input_data_node = max_pool_node->input_value(0);
    const ov::Shape max_pool_input_shape = max_pool_node->get_input_shape(0);

    const ov::Shape transpose_before_order = MakeTransposeOrderNCHW2NHWC(max_pool_input_shape.size());

    auto transpose_const =
        Constant::create(element::i64, ov::Shape{transpose_before_order.size()}, transpose_before_order);

    auto transpose_before = std::make_shared<Transpose>(max_pool_input_data_node, transpose_const);

    auto max_pool_new = std::make_shared<ov::intel_gna::op::GNAMaxPool>(transpose_before,
                                                                        max_pool_node->get_strides(),
                                                                        max_pool_node->get_pads_begin(),
                                                                        max_pool_node->get_pads_end(),
                                                                        max_pool_node->get_kernel(),
                                                                        max_pool_node->get_rounding_type(),
                                                                        max_pool_node->get_auto_pad());

    const ov::Shape transpose_after_order = MakeTransposeOrderNHWC2NCHW(max_pool_new->get_output_shape(0).size());

    auto transpose_after = std::make_shared<Transpose>(
        max_pool_new,
        Constant::create(element::i64, ov::Shape{transpose_after_order.size()}, transpose_after_order));

    ov::copy_runtime_info(max_pool_node, {transpose_before, transpose_const, max_pool_new, transpose_after});

    ov::replace_output_update_name(max_pool->output(0), transpose_after->output(0));

    return true;
}

}  // namespace SubstituteGNAMaxPoolNS

// ----------------------------------------------------------------------------

ov::intel_gna::pass::SubstituteGNAConvolution::SubstituteGNAConvolution() {
    MATCHER_SCOPE(SubstituteGNAConvolution);

    auto convolution = wrap_type<Convolution>();

    matcher_pass_callback callback = [=](Matcher& m) {
        auto convolution_node = std::dynamic_pointer_cast<Convolution>(m.get_match_root());
        if (!convolution_node) {
            return false;
        }

        return SubstituteGNAConvolutionNS::DoTransformation(convolution_node);
    };

    auto m = std::make_shared<Matcher>(convolution, matcher_name);
    this->register_matcher(m, callback);
}

ov::intel_gna::pass::SubstituteGNAMaxPool::SubstituteGNAMaxPool() {
    MATCHER_SCOPE(SubstituteGNAMaxPool);

    auto max_pool = wrap_type<ov::op::v1::MaxPool>();

    matcher_pass_callback callback = [=](Matcher& m) {
        auto max_pool_node = std::dynamic_pointer_cast<ov::op::v1::MaxPool>(m.get_match_root());
        if (!max_pool_node) {
            return false;
        }

        return SubstituteGNAMaxPoolNS::DoTransformation(max_pool_node);
    };

    auto m = std::make_shared<Matcher>(max_pool, matcher_name);
    this->register_matcher(m, callback);
}

bool ov::intel_gna::pass::TransposeNCHW::run_on_model(const std::shared_ptr<Model>& function) {
    RUN_ON_FUNCTION_SCOPE(TransposeNCHW);

    ov::pass::Manager manager(get_pass_config());
    manager.register_pass<ov::intel_gna::pass::SubstituteGNAConvolution>();
    manager.register_pass<ov::intel_gna::pass::SubstituteGNAMaxPool>();
    manager.run_passes(function);

    return false;
}
