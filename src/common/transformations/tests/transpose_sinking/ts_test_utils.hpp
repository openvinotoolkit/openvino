// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common_test_utils/ngraph_test_utils.hpp"
#include "gtest/gtest.h"
#include "openvino/frontend/manager.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino/pass/manager.hpp"

namespace transpose_sinking {
namespace testing {
namespace utils {

using NodePtr = std::shared_ptr<ov::Node>;

class IFactory {
public:
    explicit IFactory(const std::string& type_name) : type_name_(type_name) {}
    virtual ~IFactory() = default;
    virtual NodePtr create(const ov::OutputVector& parent_nodes) const = 0;

    const std::string& getTypeName() const {
        return type_name_;
    }

private:
    const std::string type_name_;
};
using FactoryPtr = std::shared_ptr<IFactory>;

class IPassFactory {
public:
    explicit IPassFactory(const std::string& type_name) : type_name_(type_name) {}
    virtual ~IPassFactory() = default;
    virtual void registerPass(ov::pass::Manager& pass_manager) const = 0;
    const std::string& getTypeName() const {
        return type_name_;
    }

private:
    const std::string type_name_;
};

template <typename PassT>
class PassFactory : public IPassFactory {
public:
    explicit PassFactory(const std::string& type_name) : IPassFactory(type_name) {}
    void registerPass(ov::pass::Manager& pass_manager) const override {
        pass_manager.register_pass<PassT>();
    }
};
using PassFactoryPtr = std::shared_ptr<IPassFactory>;
#define CREATE_PASS_FACTORY(pass_name) std::make_shared<PassFactory<ov::pass::transpose_sinking::pass_name>>(#pass_name)

std::string to_string(const ov::Shape& shape);
ov::OutputVector set_transpose_for(const std::vector<size_t>& idxs, const ov::OutputVector& out_vec);
ov::OutputVector set_gather_for(const std::vector<size_t>& idxs, const ov::OutputVector& out_vec);
std::shared_ptr<ov::Node> create_main_node(const ov::OutputVector& inputs, size_t num_ops, const FactoryPtr& creator);
ov::ParameterVector filter_parameters(const ov::OutputVector& out_vec);

std::shared_ptr<ov::Node> parameter(ov::element::Type el_type, const ov::PartialShape& ps);
template <class T>
std::shared_ptr<ov::Node> constant(ov::element::Type el_type, const ov::Shape& shape, const std::vector<T>& value) {
    return ov::opset10::Constant::create<T>(el_type, shape, value);
}

}  // namespace utils
}  // namespace testing
}  // namespace transpose_sinking