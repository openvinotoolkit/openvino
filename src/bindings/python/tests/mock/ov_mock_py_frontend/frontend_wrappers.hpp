// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/onnx/frontend.hpp"
#include "openvino/frontend/paddle/frontend.hpp"
#include "openvino/frontend/tensorflow/frontend.hpp"
#include "visibility.hpp"

// TODO: create Wrapper for ONNX. How to check that converter is actually registered?
// m_op_translators is some internal entity for ONNX FrontEnd
/*class MOCK_API FrontEndWrapperONNX : public ov::frontend::onnx::FrontEnd {
public:
    void add_extension(const std::shared_ptr<ov::Extension>& extension) override {
        FrontEnd::add_extension(extension);
    }

    bool check_conversion_extension_registered(const std::string& name) {
        return m_op_translators.find(name) != m_op_translators.end();
    }
};*/

class MOCK_API FrontEndWrapperTensorflow : public ov::frontend::tensorflow::FrontEnd {
public:
    FrontEndWrapperTensorflow() = default;
    void add_extension(const std::shared_ptr<ov::Extension>& extension) override {
        FrontEnd::add_extension(extension);
    }

    bool check_conversion_extension_registered(const std::string& name) {
        return m_op_translators.find(name) != m_op_translators.end();
    }
};

class MOCK_API FrontEndWrapperPaddle : public ov::frontend::paddle::FrontEnd {
public:
    FrontEndWrapperPaddle() = default;
    void add_extension(const std::shared_ptr<ov::Extension>& extension) override {
        FrontEnd::add_extension(extension);
    }

    bool check_conversion_extension_registered(const std::string& name) {
        return m_op_translators.find(name) != m_op_translators.end();
    }
};
