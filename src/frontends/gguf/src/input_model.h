#pragma once

#include <openvino/frontend/input_model.hpp>

#include "openvino/frontend/gguf/decoder.h"

namespace ov {
namespace frontend {
namespace gguf {

class FrontEnd;
class GgufDecoder;
using ov::frontend::gguf::GgufDecoder;

class InputModel : public ov::frontend::InputModel {
    friend class ::ov::frontend::gguf::FrontEnd;

public:
    explicit InputModel(const std::shared_ptr<GgufDecoder>& gdecoder);

    const std::shared_ptr<GgufDecoder>& get_model_decoder() const;

private:
    std::shared_ptr<GgufDecoder> m_decoder;
};

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
