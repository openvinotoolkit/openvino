#include "openvino/frontend/gguf/frontend.h"

#include "input_model.h"
#include "op_table.h"
#include "translate_session.h"

namespace ov {
namespace frontend {
namespace gguf {

FrontEnd::FrontEnd() {}

std::shared_ptr<Model> FrontEnd::convert(const InputModel::Ptr & model, bool naive) {
    auto gguf_model = std::dynamic_pointer_cast<gguf::InputModel>(model);
    FRONT_END_GENERAL_CHECK(gguf_model, "Invalid input model");
    std::shared_ptr<Model> converted_model;
    const auto & supported_ops = get_supported_ops();
    {
        TranslateSession translate_session(model, supported_ops, naive);
        converted_model = translate_session.get_converted_model();
    }
    return converted_model;
}

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
