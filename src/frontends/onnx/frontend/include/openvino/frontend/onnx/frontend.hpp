// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/frontend/frontend.hpp>
#include <openvino/frontend/telemetry_extension.hpp>

#ifdef OPENVINO_STATIC_LIBRARY
#    define ONNX_FRONTEND_API
#    define ONNX_FRONTEND_C_API
#else
#    ifdef onnx_ov_frontend_EXPORTS
#        define ONNX_FRONTEND_API   OPENVINO_CORE_EXPORTS
#        define ONNX_FRONTEND_C_API OPENVINO_EXTERN_C OPENVINO_CORE_EXPORTS
#    else
#        define ONNX_FRONTEND_API   OPENVINO_CORE_IMPORTS
#        define ONNX_FRONTEND_C_API OPENVINO_EXTERN_C OPENVINO_CORE_IMPORTS
#    endif  // onnx_ov_frontend_EXPORTS
#endif      // OPENVINO_STATIC_LIBRARY

namespace ov {
namespace frontend {
namespace onnx {

class ONNX_FRONTEND_API FrontEnd : public ov::frontend::FrontEnd {
public:
    std::shared_ptr<ov::Model> convert(InputModel::Ptr model) const override;
    void convert(std::shared_ptr<ov::Model> partially_converted) const override;
    std::shared_ptr<ov::Model> decode(InputModel::Ptr model) const override;
    std::string get_name() const override;
    bool supported_impl(const std::vector<ov::Any>& variants) const override;
    void add_extension(const std::shared_ptr<ov::Extension>& extension) override;

protected:
    InputModel::Ptr load_impl(const std::vector<ov::Any>& params) const override;

private:
    std::shared_ptr<TelemetryExtension> m_telemetry;
    //std::shared_ptr<ProgressReporterExtension> m_progress_reporter;
};

}  // namespace onnx
}  // namespace frontend
}  // namespace ov
