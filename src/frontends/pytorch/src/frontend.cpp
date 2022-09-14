#include "openvino/frontend/pytorch/frontend.hpp"

#include <exception>
#include <limits>
#include <map>
#include <memory>
#include <string>

#include "exception.hpp"
#include "input_model.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "op_table.hpp"
#include "openvino/frontend/exception.hpp"
#include "pt_framework_node.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {

std::shared_ptr<Model> FrontEnd::convert(const ov::frontend::InputModel::Ptr& model) const {
    try {
        // std::cerr << "[   HERE   ]\n";
        auto pytorch_model = std::dynamic_pointer_cast<pytorch::InputModel>(model);
        // TODO: Remove this super-hack, tensor_map should be local for each conversion activity, see more info where
        // tensor_map is defined now
        auto model = convert_pytorch_model(pytorch_model->m_model);

        // TODO: Propose better solution for the next code block
        // Usually if nn.Module.forward is given as a source model for conversion, there is the first Parameter
        // that represents original `self` argument in forward(self, ...). `self` shouldn't play any role in model
        // inference if model is completelly frozed and all methods are inlined. So we check if it doesn't have any
        // consumers in the finally converted model and remove this parameter. This parameter should have index 0.
        if (model->get_parameters().size() > 0) {
            auto self = model->get_parameters()[0];
            if (self->output(0).get_target_inputs().empty()) {
                // There is no consumers: safe to remove
                std::cout << "[ WARNING ] Removing parameter[0] in converted Pytorch model, because it is never "
                             "used and treated as `self`\n";
                model->remove_parameter(self);
            } else {
                std::cout << "[ WARNING ] Couldn't remove parameter[0] in converted Pytorch model\n";
            }
        }

        return model;
    } catch (const std::runtime_error& e) {
        std::cerr << "[ ERROR ] Error while converting pytorch model: " << e.what() << "\n";
        std::cerr << "Rethrowing. Misleading error message from pybind11 may come next. TODO.";
        throw;
    }
}

bool FrontEnd::supported_impl(const std::vector<ov::Any>& variants) const {
    // std::cout << "[  ----- DEBUG ------ ] supported_impl with " << variants.size() << " arguments\n";
    return false;
}

ov::frontend::InputModel::Ptr FrontEnd::load_impl(const std::vector<ov::Any>& variants) const {
    // std::cout << "[  ----- DEBUG -----  ] load_impl with " << variants.size() << " parameters\n";
    if (variants.size() != 1) {
        throw std::runtime_error("Pytorch frontend supports exactly one parameter in model representation, got " +
                                 std::to_string(variants.size()) + "instead.");
    }
    auto decoder = variants[0].as<std::shared_ptr<Decoder>>();
    // std::cout << "Recognized decoder: " << decoder << "\n";
    return std::make_shared<pytorch::InputModel>(decoder);
}

}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
