#pragma once

#include <torch/csrc/jit/ir/ir.h>
#include <ngraph/ngraph.hpp>

namespace torch {
namespace jit {
namespace fuser {
namespace openvino {

using TensorArgs = std::vector<at::Tensor>;

std::shared_ptr<ngraph::Function> convert(std::shared_ptr<Graph> graph, const TensorArgs& rt_inputs = {});

} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch
