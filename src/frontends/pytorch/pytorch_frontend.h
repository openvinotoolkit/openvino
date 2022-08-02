#pragma once

#include <torch/csrc/jit/ir/ir.h>
#include <ngraph/ngraph.hpp>

namespace ov {
namespace frontend {
namespace pytorch {

using TensorArgs = std::vector<at::Tensor>;

std::shared_ptr<ngraph::Function> convert(std::shared_ptr<Graph> graph, const TensorArgs& rt_inputs = {});

} // namespace pytorch
} // namespace frontend
} // namespace ov

