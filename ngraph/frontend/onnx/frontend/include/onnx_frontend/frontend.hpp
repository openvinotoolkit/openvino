// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <frontend_manager/frontend.hpp>

#ifdef onnx_ngraph_frontend_EXPORTS
#    define ONNX_FRONTEND_API NGRAPH_HELPER_DLL_EXPORT
#else
#    define ONNX_FRONTEND_API NGRAPH_HELPER_DLL_IMPORT
#endif

namespace ngraph {
namespace frontend {
class ONNX_FRONTEND_API FrontEndONNX : public FrontEnd {
public:
    std::shared_ptr<ngraph::Function> convert(InputModel::Ptr model) const override;
    void convert(std::shared_ptr<ngraph::Function> partially_converted) const override;
    std::shared_ptr<ngraph::Function> decode(InputModel::Ptr model) const override;
    std::string get_name() const override;
    bool supported_impl(const std::vector<std::shared_ptr<Variant>>& variants) const override;

protected:
    InputModel::Ptr load_impl(const std::vector<std::shared_ptr<Variant>>& params) const override;
};

}  // namespace frontend

}  // namespace ngraph
