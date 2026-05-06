// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/filter_precision_attribute.hpp"

#include <algorithm>

#include "low_precision/fake_quantize_decomposition.hpp"
#include "low_precision/network_helper.hpp"
#include "low_precision/rt_info/precisions_attribute.hpp"
#include "openvino/opsets/opset1_decl.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov {
namespace pass {
namespace low_precision {

FilterPrecisionAttribute::FilterPrecisionAttribute() {
    auto matcher = pattern::wrap_type<opset1::FakeQuantize>();

    ov::graph_rewrite_callback callback = [](pattern::Matcher& m) {
        auto fq = ov::as_type_ptr<opset1::FakeQuantize>(m.get_match_root());
        if (!fq) {
            return false;
        }

        if (NetworkHelper::isConstantPath(fq)) {
            return false;
        }

        auto attribute = getAttributeFromOutput<PrecisionsAttribute>(fq->output(0));
        if (attribute.empty()) {
            return false;
        }

        const auto& precisions = attribute.as<PrecisionsAttribute>().value();
        if (precisions.size() <= 1ul) {
            return false;
        }

        // Resolve multi-precision attribute to a single precision based on FQ output ranges
        getDataPrecisionByOutputPort(fq);
        return false;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(matcher, "FilterPrecisionAttribute");
    this->register_matcher(m, callback);
}

// get precision details, depends on:
// 1. FakeQuantize operation parameters (QuantizationDetails::getDetails & LayerTransformation::getPrecisionDetails)
// 2. Precisions on port
DataPrecision FilterPrecisionAttribute::getDataPrecisionByOutputPort(std::shared_ptr<ov::op::v0::FakeQuantize> layer) {
    const size_t levels = layer->get_levels();
    std::vector<float> outputLowValues, outputHighValues;
    if (!fq_decomposition::getOutputRanges(layer, outputLowValues, outputHighValues)) {
        return DataPrecision();
    }

    auto precisionsAttribute = getAttributeFromOutput<PrecisionsAttribute>(layer->output(0));
    if (precisionsAttribute.empty()) {
        // TODO: explore this case in more details:
        // 1. we should not be here
        assert(true);

        // 2. not possible to get optimal precision by decomposed FakeQuantize
        LayerTransformation::PrecisionDetails precisionDetailsAtOutputIntervals =
            LayerTransformation::getPrecisionDetails(levels, outputLowValues, outputHighValues);

        return DataPrecision(precisionDetailsAtOutputIntervals.precision,
                             DataPrecision::getMinValue(precisionDetailsAtOutputIntervals.precision, levels),
                             DataPrecision::getMaxValue(precisionDetailsAtOutputIntervals.precision, levels),
                             precisionDetailsAtOutputIntervals.hasZeroPoint);
    }

    const auto& precisions = precisionsAttribute.as<PrecisionsAttribute>().value();
    std::vector<element::Type> precisionsForLevels{};
    switch (levels) {
    case low_precision::levels::int16:
    case low_precision::levels::int16_narrow_range:
        precisionsForLevels = {element::u16, element::i16};
        break;
    case low_precision::levels::int32:
    case low_precision::levels::int32_narrow_range:
        precisionsForLevels = {element::u32, element::i32};
        break;
    default:
        precisionsForLevels = {element::u8, element::i8};
    }
    const auto resultPrecisions = NetworkHelper::precisionIntersection(precisions, precisionsForLevels);
    if (resultPrecisions.empty()) {
        return DataPrecision();
    }

    ov::element::Type precision;
    if (resultPrecisions.size() > 1ul) {
        LayerTransformation::PrecisionDetails precisionDetailsAtOutputIntervals =
            LayerTransformation::getPrecisionDetails(levels, outputLowValues, outputHighValues);
        const auto foundIt = std::find(resultPrecisions.begin(),
                                       resultPrecisions.end(),
                                       precisionDetailsAtOutputIntervals.precision);

        if (foundIt == resultPrecisions.end()) {
            precision = *resultPrecisions.begin();
        } else {
            precision = precisionDetailsAtOutputIntervals.precision;
        }

        // update shared attribute to affect all operations in subgraph
        precisionsAttribute.as<PrecisionsAttribute>().value() = {precision};
    } else {
        precision = *resultPrecisions.begin();
    }

    return fq_decomposition::makeDataPrecision(precision, levels, outputLowValues, outputHighValues);
}

}  // namespace low_precision
}  // namespace pass
}  // namespace ov
