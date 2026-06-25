#include "input_model.h"

#include "openvino/frontend/gguf/decoder.h"

namespace ov {
namespace frontend {
namespace gguf {

InputModel::InputModel(const std::shared_ptr<GgufDecoder> & gdecoder) : m_decoder(gdecoder) {}

const std::shared_ptr<GgufDecoder> & InputModel::get_model_decoder() const {
    return m_decoder;
}

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
