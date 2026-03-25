/*
 * Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 *
 * This file is part of the OpenVINO project.
 *
 * For more information, please visit https://openvino.ai/
 */

#ifndef OPENVINO_FRONTENDS_GLM_OCR_GLM_OCR_FRONTEND_HPP
#define OPENVINO_FRONTENDS_GLM_OCR_GLM_OCR_FRONTEND_HPP

#include <openvino/frontend.hpp>

namespace ov {
namespace frontend {

class GLMOCRFrontEnd : public FrontEnd {
public:
    GLMOCRFrontEnd();
    ~GLMOCRFrontEnd() override;

    void loadModel(const std::string& modelPath) override;
    void parseGraph() override;
    ModelPtr getModel() override;

private:
    // Private members
};

} // namespace frontend
} // namespace ov

#endif
