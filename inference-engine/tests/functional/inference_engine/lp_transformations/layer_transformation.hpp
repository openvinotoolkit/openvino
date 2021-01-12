// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common_test_utils/test_common.hpp"
#include "low_precision/rt_info/intervals_alignment_attribute.hpp"
#include "low_precision/rt_info/precisions_attribute.hpp"
#include "low_precision/layer_transformation.hpp"
#include "low_precision/transformation_context.hpp"
#include "low_precision/network_helper.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"

using namespace ngraph;

typedef std::tuple<
    element::Type,
    Shape,
    pass::low_precision::LayerTransformation::Params> LayerTransformationParams;

class LayerTransformation : public CommonTestUtils::TestsCommon {
public:
    static pass::low_precision::LayerTransformation::Params createParamsU8U8();
    static pass::low_precision::LayerTransformation::Params createParamsU8I8();
    static pass::low_precision::LayerTransformation::Params createParamsI8I8();
    static pass::low_precision::LayerTransformation::Params createParamsU8I8AndI8();

    static std::string toString(const pass::low_precision::LayerTransformation::Params& params);

    static std::string getTestCaseNameByParams(
        const element::Type& type,
        const Shape& shape,
        const pass::low_precision::LayerTransformation::Params& params);

    static builder::subgraph::DequantizationOperations toDequantizationOperations(
        const pass::low_precision::FakeQuantizeDequantization& dequantization);

    template <class Operation>
    static NodeVector get(std::shared_ptr<ngraph::Function> function) {
        NodeVector foundNodes;
        NodeVector nodes = function->get_ordered_ops();

        for (auto& node : nodes) {
            if (ngraph::is_type<Operation>(node)) {
                foundNodes.push_back(node);
            }
        }
        return foundNodes;
    }

    static bool checkIfOutputAttributesAreEqual(const NodeVector& nodes, float intervalLow, float intervalHigh) {
        for (size_t nodeIndex = 0ul; nodeIndex < nodes.size(); nodeIndex++) {
            auto& rt = nodes[nodeIndex]->get_rt_info();
            for (auto& it : rt) {
                auto reference = std::dynamic_pointer_cast<VariantWrapper<std::shared_ptr<IntervalsAlignmentAttribute>>>(it.second);
                assert(reference != nullptr);
                if ((reference->get()->sharedValue->intervalLow != intervalLow) &&
                    (reference->get()->sharedValue->intervalHigh != intervalHigh)) {
                    return false;
                }
            }
        }

        return true;
    }

    static bool compare(
        const std::shared_ptr<IntervalsAlignmentAttribute>& value1,
        const std::shared_ptr<IntervalsAlignmentAttribute>& value2) {
        if ((value1->sharedValue->intervalLow != value2->sharedValue->intervalLow) ||
            (value1->sharedValue->intervalHigh != value2->sharedValue->intervalHigh)) {
            return false;
        }
        return true;
    }

    template <class Operation>
    static bool checkIfOutputAttributesAreEqual(const NodeVector& actualNodes, const NodeVector& referenceNodes) {
        if (actualNodes.size() != referenceNodes.size()) {
            return false;
        }

        for (size_t nodeIndex = 0ul; nodeIndex < actualNodes.size(); nodeIndex++) {
            auto& actualRt = actualNodes[nodeIndex]->get_rt_info();
            auto& referenceRt = referenceNodes[nodeIndex]->get_rt_info();
            if (actualRt.size() != referenceRt.size()) {
                return false;
            }

            for (auto& actualIt : actualRt) {
                auto referenceIt = referenceRt.find(actualIt.first);
                if (referenceIt == referenceRt.end()) {
                    return false;
                }

                auto reference = std::dynamic_pointer_cast<VariantWrapper<Operation>>(referenceIt->second);
                auto actual = std::dynamic_pointer_cast<VariantWrapper<Operation>>(actualIt.second);
                if ((actual != nullptr) && (reference != nullptr)) {
                    if (!compare(reference->get(), actual->get())) {
                        return false;
                    }
                }
            }
        }

        return true;
    }

    template <class Attribute>
    static bool checkIfOutputAttributesAreTheSame(const NodeVector& nodes) {
        Variant* first = nullptr;
        for (auto node : nodes) {
            for (auto output : node->outputs()) {
                auto& rt = output.get_rt_info();
                const std::string& name = VariantWrapper<Attribute>::type_info.name;
                auto it = rt.find(name);
                if (it == rt.end()) {
                    return false;
                }

                auto value = it->second;
                if (first == nullptr) {
                    first = value.get();
                } else if (value.get() != first) {
                    return false;
                }
            }
        }
        return true;
    }

    template <class Attribute>
    static bool checkIfOutputAttributesSharedValuesAreTheSame(const NodeVector& nodes) {
        std::shared_ptr<Variant> first = nullptr;
        for (auto node : nodes) {
            for (auto output : node->outputs()) {
                auto value = ngraph::pass::low_precision::getAttributeFromOutput<Attribute>(output);
                if (first == nullptr) {
                    first = value;
                } else {
                    const auto sharedValue1 = std::dynamic_pointer_cast<ngraph::VariantWrapper<Attribute>>(value)->get()->sharedValue;
                    const auto sharedValue2 = std::dynamic_pointer_cast<ngraph::VariantWrapper<Attribute>>(first)->get()->sharedValue;
                    if (sharedValue1 != sharedValue2) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    template <class Attribute>
    static bool checkIfAttributesSharedValuesAreTheSame(const NodeVector& nodes) {
        std::shared_ptr<Variant> first = nullptr;
        for (auto node : nodes) {
            auto value = ngraph::pass::low_precision::getAttribute<Attribute>(node);
            if (value == nullptr) {
                return false;
            }

            if (first == nullptr) {
                first = value;
            } else {
                const auto sharedValue1 = std::dynamic_pointer_cast<ngraph::VariantWrapper<Attribute>>(value)->get()->sharedValue;
                const auto sharedValue2 = std::dynamic_pointer_cast<ngraph::VariantWrapper<Attribute>>(first)->get()->sharedValue;
                if (sharedValue1 != sharedValue2) {
                    return false;
                }
            }
        }
        return true;
    }

    template <class Attribute>
    static bool checkIfAttributesAreTheSame(const NodeVector& nodes) {
        Variant* first = nullptr;
        for (auto node : nodes) {
            auto value = ngraph::pass::low_precision::getAttribute<Attribute>(node);
            if (value == nullptr) {
                return false;
            }

            if (first == nullptr) {
                first = value.get();
            } else if (value.get() != first) {
                return false;
            }
        }
        return true;
    }

protected:
    std::shared_ptr<Function> actualFunction;
    std::shared_ptr<Function> referenceFunction;
};
