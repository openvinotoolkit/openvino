#include "openvino/frontend/pytorch/frontend.hpp"

#include <exception>
#include <limits>
#include <map>
#include <memory>
#include <string>

#include "exception.hpp"
#include "input_model.hpp"
#include "op_table.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "pt_framework_node.hpp"
#include "transforms.hpp"
#include "transforms/aten_cat_replacer.hpp"
#include "transforms/prim_list_unpack_replacer.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {

std::shared_ptr<Model> FrontEnd::convert(const ov::frontend::InputModel::Ptr& model) const {
    auto converted_model = convert_partially(model);
    normalize(converted_model);
    std::set<std::string> unconverted_ops_types;
    for (const auto& node : converted_model->get_ordered_ops()) {
        if (const auto& fw_node = ov::as_type_ptr<ov::frontend::pytorch::PtFrameworkNode>(node)) {
            auto op_type = fw_node->get_decoder()->get_op_type();
            unconverted_ops_types.insert(op_type);
        }
    }
    std::stringstream ops_str;
    for (auto&& op_type : unconverted_ops_types) {
        ops_str << op_type << "\n";
    }
    FRONT_END_OP_CONVERSION_CHECK(unconverted_ops_types.size() == 0,
                                  "Model wasn't fully converted. Unconverted operation types:\n" + ops_str.str());
    return converted_model;
}

void FrontEnd::convert(const std::shared_ptr<Model>& partiallyConverted) const {
    FRONT_END_NOT_IMPLEMENTED(convert);
}

std::shared_ptr<Model> FrontEnd::convert_partially(const ov::frontend::InputModel::Ptr& model) const {
    try {
        auto pytorch_model = std::dynamic_pointer_cast<pytorch::InputModel>(model);
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
                // std::cout << "[ WARNING ] Removing parameter[0] in converted Pytorch model, because it is never "
                //             "used and treated as `self`\n";
                model->remove_parameter(self);
            } else {
                std::cout << "[ WARNING ] Couldn't remove parameter[0] in converted Pytorch model\n";
            }
        }
        return model;
    } catch (const std::runtime_error& e) {
        std::cerr << "[ ERROR ] Unexpected error while converting pytorch model: " << e.what() << "\n";
        std::cerr << "Rethrowing. Misleading error message from pybind11 may come next. TODO.";
        throw;
    }
}

std::shared_ptr<Model> FrontEnd::decode(const InputModel::Ptr& model) const {
    FRONT_END_NOT_IMPLEMENTED(decode);
}

void FrontEnd::normalize(const std::shared_ptr<ov::Model>& model) const {
    ov::pass::Manager manager;

    manager.register_pass<ov::frontend::pytorch::pass::AtenCatToConcat>();
    manager.register_pass<ov::frontend::pytorch::pass::PrimListUnpackReplacer>();

    manager.run_passes(model);

    apply_pytorch_conversion_transforms(model);
}

void FrontEnd::add_extension(const std::shared_ptr<ov::Extension>& extension) {
    FRONT_END_NOT_IMPLEMENTED(add_extension);
}

bool FrontEnd::supported_impl(const std::vector<ov::Any>& variants) const {
    return false;
}

ov::frontend::InputModel::Ptr FrontEnd::load_impl(const std::vector<ov::Any>& variants) const {
    if (variants.size() != 1) {
        throw std::runtime_error("Pytorch frontend supports exactly one parameter in model representation, got " +
                                 std::to_string(variants.size()) + " instead.");
    }
    auto decoder = variants[0].as<std::shared_ptr<Decoder>>();
    return std::make_shared<pytorch::InputModel>(decoder);
}

}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
