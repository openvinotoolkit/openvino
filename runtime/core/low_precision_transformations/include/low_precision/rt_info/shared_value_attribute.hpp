// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <unordered_set>
#include <vector>

#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>

#include <low_precision/lpt_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

template <class SharedAttributeType>
class LP_TRANSFORMATIONS_API SharedValue;

template <class SharedValueType>
class LP_TRANSFORMATIONS_API SharedValueAttribute {
public:
    SharedValueAttribute() : sharedValue(std::make_shared<SharedValueType>()) {}
    virtual ~SharedValueAttribute() = default;
    std::shared_ptr<SharedValueType> sharedValue;
    std::string get_string() {
        std::stringstream ss;

        const size_t rawPointer = (size_t)this;
        ss << rawPointer << ": ";

        const size_t sharedValueRawPointer = (size_t)sharedValue.get();
        ss << "sharedValue: " << sharedValueRawPointer;

        bool firstAttribute = true;
        ss << ", attributes: {";
        for (auto& attributeWeakPtr : sharedValue->attributes) {
            auto attribute = attributeWeakPtr.lock();
            if (attribute == nullptr) {
                continue;
            }

            if (!firstAttribute) {
                ss << ", ";
            }
            ss << (size_t)attribute.get();
            firstAttribute = false;
        }
        ss << "}, ";
        return ss.str();
    }
};

template <class SharedValueAttributeType>
class LP_TRANSFORMATIONS_API SharedValue {
public:
    virtual ~SharedValue() = default;
    std::vector<std::weak_ptr<SharedValueAttributeType>> attributes;
};
