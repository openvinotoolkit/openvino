// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <unordered_set>

#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>

#include <transformations_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

template <class SharedAttributeType>
class SharedValue;

template <class SharedValueType>
class SharedAttribute {
public:
    SharedAttribute() : sharedValue(std::make_shared<SharedValueType>()) {};
    virtual ~SharedAttribute() = default;
    std::shared_ptr<SharedValueType> sharedValue;
};

template <class SharedAttributeType>
class SharedValue {
public:
    virtual ~SharedValue() = default;
    std::vector<std::shared_ptr<SharedAttributeType>> attributes;
};
