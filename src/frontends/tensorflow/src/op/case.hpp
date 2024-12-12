#ifndef CASE_HPP
#define CASE_HPP

#include "openvino/frontend/tensorflow/node_context.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_case_op(const ov::frontend::tensorflow::NodeContext& node);

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov

#endif  // CASE_HPP
