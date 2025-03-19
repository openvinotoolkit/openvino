// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common_test_utils/test_common.hpp"
#include "low_precision/rt_info/intervals_alignment_attribute.hpp"
#include "low_precision/rt_info/precisions_attribute.hpp"
#include "low_precision/layer_transformation.hpp"
#include "low_precision/network_helper.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"

using namespace ov;

typedef std::tuple<
    element::Type,
    Shape,
    ov::pass::low_precision::LayerTransformation::Params> LayerTransformationParams;

struct TestTransformationParams {
    TestTransformationParams(
        bool updatePrecisions = true,
        std::vector<element::Type> precisionsOnActivations = { element::u8, element::i8 },
        std::vector<element::Type> precisionsOnWeights = { element::i8 },
        bool supportAsymmetricQuantization = true,
        element::Type deqPrecision = element::f32,
        bool deconvolutionSpecificChannelsRatio = false,
        std::vector<ov::element::Type> defaultPrecisions = { element::u8, element::i8 });

    TestTransformationParams& setDeqPrecision(const element::Type deqPrecision);
    TestTransformationParams& setUpdatePrecisions(const bool updatePrecisions);
    TestTransformationParams& setSupportAsymmetricQuantization(const bool supportAsymmetricQuantization);
    TestTransformationParams& setPrecisionsOnActivations(const std::vector<element::Type>& precisionsOnActivations);
    TestTransformationParams& setPrecisionsOnWeights(const std::vector<element::Type>& precisionsOnWeights);
    TestTransformationParams& setDeconvolutionSpecificChannelsRatio(const bool deconvolutionSpecificChannelsRatio);
    TestTransformationParams& setDefaultPrecisions(const std::vector<element::Type>& defaultPrecisions);

    static ov::pass::low_precision::LayerTransformation::Params toParams(const TestTransformationParams& params);

    bool updatePrecisions;
    std::vector<element::Type> precisionsOnActivations;
    std::vector<element::Type> precisionsOnWeights;
    bool supportAsymmetricQuantization;
    element::Type deqPrecision;
    bool deconvolutionSpecificChannelsRatio;
    std::vector<element::Type> defaultPrecisions;
};

class LayerTransformation : public ov::test::TestsCommon {
public:
    static TestTransformationParams createParamsU8U8();
    static TestTransformationParams createParamsU8I8();
    static TestTransformationParams createParamsI8I8();
    static TestTransformationParams createParamsU8I8AndI8();

    static std::string toString(const TestTransformationParams& params);

    static std::string getTestCaseNameByParams(
        const ov::element::Type& type,
        const ov::PartialShape& shape,
        const TestTransformationParams& params);

    static ov::builder::subgraph::DequantizationOperations toDequantizationOperations(
        const ov::pass::low_precision::FakeQuantizeDequantization& dequantization);

    static bool allNamesAreUnique(const std::shared_ptr<ov::Model>& model);

    template <class Operation>
    static NodeVector get(std::shared_ptr<ov::Model> model) {
        NodeVector foundNodes;
        NodeVector nodes = model->get_ordered_ops();

        for (auto& node : nodes) {
            if (ov::is_type<Operation>(node)) {
                foundNodes.push_back(node);
            }
        }
        return foundNodes;
    }

    static bool checkIfOutputAttributesAreEqual(const NodeVector& nodes, float intervalLow, float intervalHigh) {
        for (size_t nodeIndex = 0ul; nodeIndex < nodes.size(); nodeIndex++) {
            auto& rt = nodes[nodeIndex]->get_rt_info();
            for (auto& it : rt) {
                auto& reference = it.second.as<ov::IntervalsAlignmentAttribute>();
                if ((reference.value().combinedInterval.low != intervalLow) &&
                    (reference.value().combinedInterval.high != intervalHigh)) {
                    return false;
                }
            }
        }

        return true;
    }

    static bool compare(
        const ov::IntervalsAlignmentAttribute& value1,
        const ov::IntervalsAlignmentAttribute& value2) {
        if ((value1.value().combinedInterval.low != value2.value().combinedInterval.low) ||
            (value1.value().combinedInterval.high != value2.value().combinedInterval.high)) {
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

                if (!referenceIt->second.empty() && !actualIt.second.empty()) {
                    if (!compare(referenceIt->second.template as<Operation>(), actualIt.second.template as<Operation>())) {
                        return false;
                    }
                }
            }
        }

        return true;
    }

    template <class Attribute>
    static bool checkIfOutputAttributesAreTheSame(const NodeVector& nodes) {
        ov::Any first;
        for (auto node : nodes) {
            for (auto output : node->outputs()) {
                auto& rt = output.get_rt_info();
                const std::string& name = Attribute::get_type_info_static();
                auto it = rt.find(name);
                if (it == rt.end()) {
                    return false;
                }

                auto value = it->second;
                if (first.empty()) {
                    first = value;
                } else if (value.template as<Attribute>().attribute != first.template as<Attribute>().attribute) {
                    return false;
                }
            }
        }
        return true;
    }

    template <class Attribute>
    static bool checkIfOutputAttributesSharedValuesAreTheSame(const NodeVector& nodes) {
        ov::Any first;
        for (auto node : nodes) {
            for (auto output : node->outputs()) {
                auto value = ov::pass::low_precision::getAttributeFromOutput<Attribute>(output);
                if (first.empty()) {
                    first = value;
                } else {
                    const auto sharedValue1 = value.template as<Attribute>().attribute->sharedValue;
                    const auto sharedValue2 = first.template as<Attribute>().attribute->sharedValue;
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
        ov::Any first;
        for (auto node : nodes) {
            auto value = ov::pass::low_precision::getAttribute<Attribute>(node);
            if (value.empty()) {
                return false;
            }

            if (first.empty()) {
                first = value;
            } else {
                const auto sharedValue1 = value.template as<Attribute>().attribute->sharedValue;
                const auto sharedValue2 = first.template as<Attribute>().attribute->sharedValue;
                if (sharedValue1 != sharedValue2) {
                    return false;
                }
            }
        }
        return true;
    }

    template <class Attribute>
    static bool checkIfAttributesAreTheSame(const NodeVector& nodes) {
        ov::Any first;
        for (auto node : nodes) {
            auto value = ov::pass::low_precision::getAttribute<Attribute>(node);
            if (value.empty()) {
                return false;
            }

            if (first.empty()) {
                first = value;
            } else if (value.template as<Attribute>().attribute != first.template as<Attribute>().attribute) {
                return false;
            }
        }
        return true;
    }

protected:
    std::shared_ptr<Model> actualFunction;
    std::shared_ptr<Model> referenceFunction;
};

template <typename T>
inline std::ostream& operator<<(std::ostream& os, const std::vector<T>& values) {
    os << "{ ";
    for (size_t i = 0; i < values.size(); ++i) {
        os << values[i];
        if (i != (values.size() - 1ul)) {
            os << ", ";
        }
    }
    os << " }";
    return os;
}
