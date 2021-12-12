// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/frontends/common/frontend.hpp>
#include <openvino/frontends/common/telemetry_extension.hpp>

namespace ov {
namespace frontend {
namespace onnx {

class FRONTEND_API FrontEnd : public ov::frontend::FrontEnd {
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
};

}  // namespace onnx
}  // namespace frontend
}  // namespace ov
