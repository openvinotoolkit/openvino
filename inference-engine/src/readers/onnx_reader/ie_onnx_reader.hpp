// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_reader.hpp>
#include "frontend_manager/frontend_manager.hpp"

namespace InferenceEngine {

class ONNXReader: public IReader {
public:
    /**
     * @brief Initializes ONNX Frontend
     */
    ONNXReader();
    /**
     * @brief Checks that reader supports format of the model
     * @param model stream with model
     * @return true if format is supported
     */
    bool supportModel(std::istream& model) const override;
    /**
     * @brief Reads the model to CNNNetwork
     * @param model stream with model
     * @param exts vector with extensions
     *
     * @return CNNNetwork
     */
    CNNNetwork read(std::istream& model, const std::vector<IExtensionPtr>& exts) const override;
    /**
     * @brief Reads the model to CNNNetwork
     * @param model stream with model
     * @param weights blob with binary data
     * @param exts vector with extensions
     *
     * @return CNNNetwork
     */
    CNNNetwork read(std::istream& model, const Blob::CPtr& weights, const std::vector<IExtensionPtr>& exts) const override {
        IE_THROW() << "ONNX reader cannot read model with weights!";
    }

    std::vector<std::string> getDataFileExtensions() const override {
        return {};
    }

    private:
        ngraph::frontend::FrontEnd::Ptr m_onnx_fe;
};

}  // namespace InferenceEngine

