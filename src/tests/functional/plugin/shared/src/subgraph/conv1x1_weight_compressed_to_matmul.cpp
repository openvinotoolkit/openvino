// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/conv1x1_weight_compressed_to_matmul.hpp"

#include <sstream>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/runtime/properties.hpp"

namespace ov {
namespace test {

namespace {
const std::string DECOR_NONE = "None";
const std::string DECOR_TRANSPOSE = "Transpose";
const std::string DECOR_RESHAPE = "Reshape";

void validate_decoration(const std::string& decoration) {
    OPENVINO_ASSERT(decoration == DECOR_NONE || decoration == DECOR_TRANSPOSE || decoration == DECOR_RESHAPE,
                    "Unsupported conv1x1 decoration: ",
                    decoration);
}
}  // namespace

std::string Conv1x1WeightCompressedToMatmulTest::getTestCaseName(
    const testing::TestParamInfo<Conv1x1WeightCompressedToMatmulParams>& obj) {
    const auto& [shape_params,
                 in_decoration,
                 out_decoration,
                 act_prec,
                 weights_prec,
                 expected_op_counts,
                 config,
                 device] = obj.param;

    std::ostringstream result;
    result << "IS=" << shape_params.data_shape << "_";
    result << "Cout=" << shape_params.channels_out << "_";
    result << "in=" << in_decoration << "_out=" << out_decoration << "_";
    result << "actPrec=" << act_prec << "_wPrec=" << weights_prec << "_config=(";
    for (const auto& configEntry : config) {
        result << configEntry.first << "_";
    }
    result << ")_device=" << device;
    return result.str();
}

void Conv1x1WeightCompressedToMatmulTest::SetUp() {
    const auto& [shape_params,
                 in_decoration,
                 out_decoration,
                 act_prec,
                 weights_prec,
                 expected_op_counts,
                 config,
                 device] = GetParam();
    targetDevice = device;
    expected_op_counts_ = expected_op_counts;
    configuration.insert(config.begin(), config.end());
    abs_threshold = 0.05f;
    rel_threshold = 0.01f;
    validate_decoration(in_decoration);
    validate_decoration(out_decoration);

    init_input_shapes({shape_params.data_shape});
    const auto& param_pshape = inputDynamicShapes[0];

    // Cin is the Convolution input channel: last dim for the Transpose input layout, dim 1 otherwise.
    const auto Cin = (in_decoration == DECOR_TRANSPOSE ? param_pshape[param_pshape.size() - 1] : param_pshape[1]).get_length();
    OPENVINO_ASSERT(in_decoration != DECOR_RESHAPE || param_pshape.size() == 2,
                    "Reshape input decoration expects a 2D [N, Cin] activation");

    auto param = std::make_shared<ov::op::v0::Parameter>(act_prec, param_pshape);

    std::shared_ptr<ov::Node> conv_input = param;
    if (in_decoration == DECOR_TRANSPOSE) {
        auto order = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{4}, {0, 3, 1, 2});
        conv_input = std::make_shared<ov::op::v1::Transpose>(param, order);
    } else if (in_decoration == DECOR_RESHAPE) {
        auto pattern = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{4}, {0, static_cast<int>(Cin), 1, 1});
        conv_input = std::make_shared<ov::op::v1::Reshape>(param, pattern, true);
    }

    const int seed = 1;
    const int up_to = weights_prec == ov::element::u2 ? 3 : (weights_prec == ov::element::i4 ? 7 : 15);
    const int start_from = weights_prec == ov::element::u2 ? 0 : 1;
    auto weights_tensor = ov::test::utils::create_and_fill_tensor(
        weights_prec,
        ov::Shape{shape_params.channels_out, static_cast<size_t>(Cin), 1, 1},
        ov::test::utils::InputGenerateData(start_from, up_to, 1, seed));
    auto weights = std::make_shared<ov::op::v0::Constant>(weights_tensor);
    auto weights_convert = std::make_shared<ov::op::v0::Convert>(weights, act_prec);
    auto scale_tensor = ov::test::utils::create_and_fill_tensor_real_distribution(
        act_prec, ov::Shape{shape_params.channels_out, 1, 1, 1}, 0.001f, 0.01f, seed);
    auto scale = std::make_shared<ov::op::v0::Constant>(scale_tensor);
    auto weights_dequant = std::make_shared<ov::op::v1::Multiply>(weights_convert, scale);

    auto conv = std::make_shared<ov::op::v1::Convolution>(conv_input,
                                                          weights_dequant,
                                                          ov::Strides{1, 1},
                                                          ov::CoordinateDiff{0, 0},
                                                          ov::CoordinateDiff{0, 0},
                                                          ov::Strides{1, 1},
                                                          ov::op::PadType::EXPLICIT);

    std::shared_ptr<ov::Node> out_node = conv;
    if (out_decoration == DECOR_TRANSPOSE) {
        auto order = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{4}, {0, 2, 3, 1});
        out_node = std::make_shared<ov::op::v1::Transpose>(conv, order);
    } else if (out_decoration == DECOR_RESHAPE) {
        auto pattern = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {0, -1});
        out_node = std::make_shared<ov::op::v1::Reshape>(conv, pattern, true);
    }

    auto result = std::make_shared<ov::op::v0::Result>(out_node);
    function = std::make_shared<ov::Model>(ov::ResultVector{result},
                                           ov::ParameterVector{param},
                                           "Conv1x1WeightCompressedToMatmul");
}

void Conv1x1WeightCompressedToMatmulTest::validate() {
    SubgraphBaseTest::validate();
    for (const auto& [layer_type, expected_count] : expected_op_counts_) {
        CheckNumberOfNodesWithType(compiledModel, layer_type, expected_count);
    }
}

}  // namespace test
}  // namespace ov
