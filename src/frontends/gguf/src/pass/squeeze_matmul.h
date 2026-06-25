#include "openvino/pass/matcher_pass.hpp"

namespace ov {
namespace frontend {
namespace gguf {
namespace pass {

class SqueezeMatmul : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::gguf::pass::SqueezeMatmul")
    SqueezeMatmul();
};

}  // namespace pass
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
