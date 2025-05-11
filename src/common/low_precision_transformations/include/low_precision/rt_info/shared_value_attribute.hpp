// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <unordered_set>
#include <vector>

#include "openvino/core/node.hpp"

#include "low_precision/lpt_visibility.hpp"

template <class T>
class LP_TRANSFORMATIONS_API SharedAttribute : public ov::RuntimeAttribute {
public:
    /**
     * @ingroup ov_transformation_common_api
     * @brief SharedValueAttribute type for shared value attributes.
     * The attribute is used for attribute SharedValue value backward propagation.
     */
    class LP_TRANSFORMATIONS_API SharedValueAttribute : public std::enable_shared_from_this<SharedValueAttribute> {
    public:
        struct LP_TRANSFORMATIONS_API SharedValue : public std::enable_shared_from_this<SharedValue> {
            SharedValue() {}
            SharedValue(const T& value) : value{value} {}
            T value = {};
            void addAttribute(std::weak_ptr<SharedValueAttribute> attribute) {
                auto attributeLocked = attribute.lock();
                if (attributeLocked == nullptr) {
                    return;
                }

                for (auto& attr : attributes) {
                    auto attrLocked = attr.lock();
                    if (attrLocked == nullptr) {
                        continue;
                    }
                    if (attributeLocked == attrLocked) {
                        return;
                    }
                }

                attributes.push_back(attribute);
            }

            std::vector<std::weak_ptr<SharedValueAttribute>>& getAttributes() {
                return attributes;
            }

        private:
            std::vector<std::weak_ptr<SharedValueAttribute>> attributes;
        };
        SharedValueAttribute() : sharedValue(std::make_shared<SharedValue>()) {}

        SharedValueAttribute(const T& value) : sharedValue{std::make_shared<SharedValue>(value)} {}

        std::shared_ptr<SharedValue> sharedValue;

        std::string get_string() {
            std::stringstream ss;

            const size_t rawPointer = (size_t)this;
            ss << rawPointer << ": ";

            const size_t sharedValueRawPointer = (size_t)sharedValue.get();
            ss << "sharedValue: " << sharedValueRawPointer;

            bool firstAttribute = true;
            ss << ", attributes: {";
            for (auto& attributeWeakPtr : sharedValue->getAttributes()) {
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

    SharedAttribute() : attribute{std::make_shared<SharedValueAttribute>()} {
        attribute->sharedValue->addAttribute(attribute);
    }
    SharedAttribute(const T& value) : attribute{std::make_shared<SharedValueAttribute>(value)} {
        attribute->sharedValue->addAttribute(attribute);
    }

    std::shared_ptr<SharedValueAttribute> attribute;

    const T& value() const {
        OPENVINO_ASSERT(attribute != nullptr, "Empty attribute");
        OPENVINO_ASSERT(attribute->sharedValue != nullptr, "Empty shared value");
        return attribute->sharedValue->value;
    }

    T& value() {
        OPENVINO_ASSERT(attribute != nullptr, "Empty attribute");
        OPENVINO_ASSERT(attribute->sharedValue != nullptr, "Empty shared value");
        return attribute->sharedValue->value;
    }
};
