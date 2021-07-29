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

struct TestTransformationParams {
    TestTransformationParams(
        bool updatePrecisions = true,
        std::vector<element::Type> precisionsOnActivations = { element::u8, element::i8 },
        std::vector<element::Type> precisionsOnWeights = { element::i8 },
        bool supportAsymmetricQuantization = true,
        element::Type deqPrecision = element::f32,
        bool support3DTensorOnActivations = true,
        bool deconvolutionSpecificChannelsRatio = false);

    TestTransformationParams& setUpdatePrecisions(const bool updatePrecisions);
    TestTransformationParams& setSupportAsymmetricQuantization(const bool supportAsymmetricQuantization);
    TestTransformationParams& setPrecisionsOnActivations(const std::vector<element::Type>& precisionsOnActivations);
    TestTransformationParams& setPrecisionsOnWeights(const std::vector<element::Type>& precisionsOnWeights);
    TestTransformationParams& setSupport3DTensorOnActivations(const bool support3DTensorOnActivations);
    TestTransformationParams& setDeconvolutionSpecificChannelsRatio(const bool deconvolutionSpecificChannelsRatio);

    static pass::low_precision::LayerTransformation::Params toParams(const TestTransformationParams& params);

    bool updatePrecisions;
    std::vector<element::Type> precisionsOnActivations;
    std::vector<element::Type> precisionsOnWeights;
    bool supportAsymmetricQuantization;
    element::Type deqPrecision;
    bool support3DTensorOnActivations;
    bool deconvolutionSpecificChannelsRatio;
};

/*
TestTransformationParams& setSupportAsymmetricQuantization(const bool supportAsymmetricQuantization) {
        this->supportAsymmetricQuantization = supportAsymmetricQuantization;
        return *this;
    }

    TestTransformationParams& setPrecisionsOnActivations(const std::vector<element::Type>& precisionsOnActivations) {
        this->precisionsOnActivations = precisionsOnActivations;
        return *this;
    }

    TestTransformationParams& setPrecisionsOnWeights(const std::vector<element::Type>& precisionsOnWeights) {
        this->precisionsOnWeights = precisionsOnWeights;
        return *this;
    }

    TestTransformationParams& setSupport3DTensorOnActivations(const bool support3DTensorOnActivations) {
        this->support3DTensorOnActivations = support3DTensorOnActivations;
        return *this;
    }

    TestTransformationParams& setDeconvolutionSpecificChannelsRatio(const bool deconvolutionSpecificChannelsRatio) {
        this->deconvolutionSpecificChannelsRatio = deconvolutionSpecificChannelsRatio;
        return *this;
    }
*/

class LayerTransformation : public CommonTestUtils::TestsCommon {
public:
    static TestTransformationParams createParamsU8U8();
    static TestTransformationParams createParamsU8I8();
    static TestTransformationParams createParamsI8I8();
    static TestTransformationParams createParamsU8I8AndI8();

    static std::string toString(const TestTransformationParams& params);

    static std::string getTestCaseNameByParams(
        const ngraph::element::Type& type,
        const ngraph::PartialShape& shape,
        const TestTransformationParams& params);

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
                if ((reference->get()->sharedValue->combinedInterval.low != intervalLow) &&
                    (reference->get()->sharedValue->combinedInterval.high != intervalHigh)) {
                    return false;
                }
            }
        }

        return true;
    }

    static bool compare(
        const std::shared_ptr<IntervalsAlignmentAttribute>& value1,
        const std::shared_ptr<IntervalsAlignmentAttribute>& value2) {
        if ((value1->sharedValue->combinedInterval.low != value2->sharedValue->combinedInterval.low) ||
            (value1->sharedValue->combinedInterval.high != value2->sharedValue->combinedInterval.high)) {
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
