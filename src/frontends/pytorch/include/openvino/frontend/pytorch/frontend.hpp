#pragma once

#include "openvino/core/any.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/frontend/extension/conversion.hpp"
#include "openvino/frontend/extension/decoder_transformation.hpp"
#include "openvino/frontend/extension/telemetry.hpp"
#include "openvino/frontend/frontend.hpp"
#include "openvino/frontend/input_model.hpp"
#include "openvino/frontend/pytorch/decoder.hpp"
#include "openvino/frontend/pytorch/visibility.hpp"
#include "openvino/frontend/visibility.hpp"

namespace ov {
namespace frontend {
namespace pytorch {

class PYTORCH_API FrontEnd : public ov::frontend::FrontEnd {
public:
    using Ptr = std::shared_ptr<FrontEnd>;

    std::shared_ptr<Model> convert(const ov::frontend::InputModel::Ptr& model) const override;

    std::string get_name() const override {
        return "pytorch";
    }

protected:
    bool supported_impl(const std::vector<ov::Any>& variants) const override;

    ov::frontend::InputModel::Ptr load_impl(const std::vector<ov::Any>& variants) const override;
};

} // namespace pytorch
} // namespace frontend
} // namespace ov

