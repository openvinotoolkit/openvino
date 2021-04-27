// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <unordered_set>
#include <vector>

#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>

#include <transformations_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

template <class SharedAttributeType>
class SharedValue;

template <class SharedValueType>
class SharedValueAttribute {
public:
    SharedValueAttribute() : sharedValue(std::make_shared<SharedValueType>()) {};
    virtual ~SharedValueAttribute() = default;
    std::shared_ptr<SharedValueType> sharedValue;
};

template <class SharedValueAttributeType>
class SharedValue {
public:
    virtual ~SharedValue() = default;
    std::vector<std::weak_ptr<SharedValueAttributeType>> attributes;
};
