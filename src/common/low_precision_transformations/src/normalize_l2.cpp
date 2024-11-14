// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/normalize_l2.hpp"

#include <string>
#include <memory>
#include <cmath>
#include <vector>

#include "itt.hpp"
#include "openvino/util/log.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/element_type_traits.hpp"

#include "low_precision/network_helper.hpp"

using namespace ov;
using namespace ov::pass;
using namespace ov::pass::low_precision;

namespace normalize_l2 {

template<typename T>
std::shared_ptr<ov::opset1::Constant> createNewScalesConst(const ov::opset1::Constant& originalConst) {
    std::vector<T> source = originalConst.cast_vector<T>();

    std::vector<T> newData(source.size());
    for (size_t i = 0; i < source.size(); ++i) {
        newData[i] = source[i] < 0 ? T{-1} : T{1};
    }

    const ov::element::Type type = originalConst.get_output_element_type(0);
    return ov::opset1::Constant::create(type, originalConst.get_shape(), newData);
}

} // namespace normalize_l2

NormalizeL2Transformation::NormalizeL2Transformation(const Params& params) : LayerTransformation(params) {
    MATCHER_SCOPE(NormalizeL2Transformation);
    auto matcher = pattern::wrap_type<ov::opset1::NormalizeL2>({ pattern::wrap_type<ov::opset1::Multiply>(), pattern::wrap_type<ov::opset1::Constant>() });

    ov::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        auto op = m.get_match_root();
        if (transformation_callback(op)) {
            return false;
        }
        return transform(*context, m);
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(matcher, matcher_name);
    this->register_matcher(m, callback);
}

bool NormalizeL2Transformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> operation) const {
    if (!LayerTransformation::canBeTransformed(context, operation)) {
        return false;
    }

    const auto dequantization = NetworkHelper::getDequantization(operation, defaultPrecisions);
    if (dequantization.subtract != nullptr) {
        return false;
    }

    const auto scalesConst = dequantization.multiplyConstant;
    if (scalesConst == nullptr) {
        return false;
    }

    // TODO: Expand transformation for all cases of axes values
    const auto axes = ov::as_type_ptr<ov::opset1::Constant>(operation->get_input_node_shared_ptr(1));
    const std::vector<int64_t> axesAcrossSpatial = { 1 };
    const std::vector<int64_t> axesByChannels = { 1, 2, 3 };

    std::vector<int64_t> axesValues = axes->cast_vector<int64_t>();
    if ((axesValues != axesAcrossSpatial) && (axesValues != axesByChannels)) {
        return false;
    }

    const Shape outputShape = scalesConst->get_shape();
    const size_t size = shape_size(outputShape);
    if (size != 1ul) {
        if (operation->get_output_partial_shape(0).size() < 2)
            return false;
        const auto channelsInterval = operation->get_output_partial_shape(0)[1];
        if (channelsInterval.is_dynamic() || static_cast<size_t>(channelsInterval.get_length()) != size) {
            return false;
        }
    }

    if (!NetworkHelper::isScalarLike(scalesConst)) {
        return false;
    }

    return true;
}

bool NormalizeL2Transformation::transform(TransformationContext &context, ov::pass::pattern::Matcher &m) {
    std::shared_ptr<Node> operation = m.get_match_root();
    if (!canBeTransformed(context, operation)) {
        return false;
    }

    auto normalize = ov::as_type_ptr<ov::opset1::NormalizeL2>(NetworkHelper::separateInStandaloneBranch(operation, defaultPrecisions));

    const auto axes = ov::as_type_ptr<ov::opset1::Constant>(normalize->get_input_node_shared_ptr(1));
    FakeQuantizeDequantization dequantization = NetworkHelper::getDequantization(normalize, defaultPrecisions);
    auto scalesConst = ov::as_type_ptr<ov::opset1::Constant>(dequantization.multiply->get_input_node_shared_ptr(1));
    if (scalesConst == nullptr) {
        scalesConst = ov::as_type_ptr<ov::opset1::Constant>(dequantization.multiply->get_input_node_shared_ptr(0));
    }

    std::shared_ptr<ov::opset1::Constant> newScalesConst;
    const auto type = scalesConst->get_output_element_type(0);
    switch (type) {
        case ov::element::Type_t::f16: {
            newScalesConst = normalize_l2::createNewScalesConst<ov::element_type_traits<ov::element::Type_t::f16>::value_type>(*scalesConst);
            break;
        }
        case ov::element::Type_t::f32: {
            newScalesConst = normalize_l2::createNewScalesConst<ov::element_type_traits<ov::element::Type_t::f32>::value_type>(*scalesConst);
            break;
        }
        default: {
            THROW_TRANSFORMATION_EXCEPTION << "unexpected element type " << type;
        }
    }

    auto newNormalize = std::make_shared<ov::op::TypeRelaxed<ov::opset1::NormalizeL2>>(
        std::vector<ov::element::Type>{ element::f32, axes->output(0).get_element_type() },
        std::vector<ov::element::Type>{deqPrecision},
        ov::op::TemporaryReplaceOutputType(dequantization.subtract == nullptr ? dequantization.data : dequantization.subtract, element::f32).get(),
        axes,
        normalize->get_eps(),
        normalize->get_eps_mode());
    NetworkHelper::copyInfo(normalize, newNormalize);

    auto newMultiply = std::make_shared<ov::op::TypeRelaxed<ov::opset1::Multiply>>(
        std::vector<ov::element::Type>{ element::f32, element::f32 },
        std::vector<ov::element::Type>{normalize->get_output_element_type(0)},
        ov::op::TemporaryReplaceOutputType(newNormalize, element::f32).get(),
        ov::op::TemporaryReplaceOutputType(newScalesConst, element::f32).get());

    NetworkHelper::insertDequantizationAfter(normalize, newMultiply, newNormalize);
    ov::copy_runtime_info({ normalize, newMultiply }, newMultiply);

    updateOutput(context, newMultiply, newNormalize);

    OPENVINO_DEBUG("LPT: done: ", newNormalize);
    return true;
}

bool NormalizeL2Transformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return false;
}
