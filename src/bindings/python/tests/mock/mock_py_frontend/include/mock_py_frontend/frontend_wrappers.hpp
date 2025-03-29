// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifdef ENABLE_OV_ONNX_FRONTEND
#    include "openvino/frontend/onnx/frontend.hpp"
#endif

#ifdef ENABLE_OV_PADDLE_FRONTEND
#    include "openvino/frontend/paddle/frontend.hpp"
#endif

#ifdef ENABLE_OV_TF_FRONTEND
#    include "openvino/frontend/tensorflow/frontend.hpp"
#endif
#include "visibility.hpp"

#ifdef ENABLE_OV_ONNX_FRONTEND
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
#endif

#ifdef ENABLE_OV_TF_FRONTEND
class MOCK_API FrontEndWrapperTensorflow : public ov::frontend::tensorflow::FrontEnd {
public:
    FrontEndWrapperTensorflow();
    void add_extension(const std::shared_ptr<ov::Extension>& extension) override;

    bool check_conversion_extension_registered(const std::string& name);
};
#endif

#ifdef ENABLE_OV_PADDLE_FRONTEND
class MOCK_API FrontEndWrapperPaddle : public ov::frontend::paddle::FrontEnd {
public:
    FrontEndWrapperPaddle();
    void add_extension(const std::shared_ptr<ov::Extension>& extension) override;

    bool check_conversion_extension_registered(const std::string& name);
};
#endif
