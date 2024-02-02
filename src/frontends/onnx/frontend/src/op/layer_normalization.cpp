// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/layer_normalization.hpp"

#include "exceptions.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/negative.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/subtract.hpp"
#include "utils/common.hpp"
#include "utils/reshape.hpp"

using namespace ov::op;
using namespace ov::op::v0;
using namespace ov::op::v1;
using namespace ov::op::v8;
using ov::Shape;

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {

ov::OutputVector layer_normalization(const Node& node) {
    const auto inputs = node.get_ng_inputs();
    const auto num_inputs = inputs.size();
    CHECK_VALID_NODE(node,
                     num_inputs == 2 || num_inputs == 3,
                     "LayerNormalization expects 2 or 3 input tensors. Got: ",
                     num_inputs);

    const auto& X = inputs.at(0);
    const auto& Scale = inputs.at(1);

    auto axis = node.get_attribute_value<std::int64_t>("axis", -1);
    double epsilon = node.get_attribute_value<double>("epsilon", 1e-5);
    int64_t stash_type_i =
        node.get_attribute_value<int64_t>("stash_type",
                                          static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT));
    element::Type stash_type = common::get_ov_element_type(stash_type_i);

    // following calculations are kept as close to the onnx\defs.cc description as possible
    auto FloatEpsilon = Constant::create(ov::element::f32, Shape{}, {epsilon});
    auto Epsilon = std::make_shared<Convert>(FloatEpsilon, stash_type);
    auto XShape = std::make_shared<ShapeOf>(X);
    auto Rank = std::make_shared<v3::ShapeOf>(XShape);
    auto Zero1D = Constant::create(ov::element::i64, {1}, {0});
    auto One1D = Constant::create(ov::element::i64, {1}, {1});
    auto Axis1D = Constant::create(ov::element::i64, {1}, {axis});
    auto PrefixShape = std::make_shared<Slice>(XShape, Zero1D, Axis1D, One1D);
    ov::Output<ov::Node> NumReducedAxes = (axis >= 0 ? std::make_shared<Subtract>(Rank, Axis1D)->output(0)
                                                     : std::make_shared<Negative>(Axis1D)->output(0));
    auto SuffixShape = std::make_shared<v3::Broadcast>(One1D, NumReducedAxes);
    auto ReducedShape = std::make_shared<Concat>(ov::OutputVector{PrefixShape, SuffixShape}, 0);

    auto X2D = util::flatten(X, static_cast<int>(axis));
    auto XU = std::make_shared<Convert>(X2D, stash_type);

    auto Mean2D = std::make_shared<ReduceMean>(XU, One1D, true);
    auto Square = std::make_shared<Multiply>(XU, XU);
    auto MeanOfSquare = std::make_shared<ReduceMean>(Square, One1D, true);
    auto SquareOfMean = std::make_shared<Multiply>(Mean2D, Mean2D);

    auto Var = std::make_shared<Subtract>(MeanOfSquare, SquareOfMean);
    auto VarPlusEpsilon = std::make_shared<Add>(Var, Epsilon);
    auto StdDev = std::make_shared<Sqrt>(VarPlusEpsilon);
    auto Deviation = std::make_shared<Subtract>(XU, Mean2D);
    auto Normalized = std::make_shared<Divide>(Deviation, StdDev);
    auto NormalizedT = std::make_shared<ConvertLike>(Normalized, X);

    auto Scale2D = util::flatten(Scale, 0);
    auto Scaled = std::make_shared<Multiply>(NormalizedT, Scale2D);
    ov::Output<ov::Node> Biased =
        (num_inputs == 3 ? std::make_shared<Add>(Scaled, util::flatten(inputs.at(2), 0))->output(0)
                         : Scaled->output(0));

    auto Y = std::make_shared<Reshape>(Biased, XShape, false);
    auto InvStdDev2D = std::make_shared<Divide>(Constant::create(stash_type, {1}, {1}), StdDev);
    auto Mean = std::make_shared<Reshape>(Mean2D, ReducedShape, false);
    auto InvStdDev = std::make_shared<Reshape>(InvStdDev2D, ReducedShape, false);

    return ov::OutputVector{Y, Mean, InvStdDev};
}

}  // namespace set_1
}  // namespace op
}  // namespace onnx_import
}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
