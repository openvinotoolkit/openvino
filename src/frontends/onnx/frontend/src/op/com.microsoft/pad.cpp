#include <vector>
#include "ngraph/op/pad.hpp"
#include "ngraph/op/constant.hpp"

#include <memory>

#include "default_opset.hpp"
#include "ngraph/builder/reshape.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/matmul.hpp"
#include "ngraph/op/multiply.hpp"
#include "onnx_import/core/null_node.hpp"

namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector microsoftPad(const Node& node) {
    OutputVector inputs{node.get_ng_inputs()};
    NGRAPH_CHECK(inputs.size() == 2 || inputs.size() == 3,
                 "MicrosoftPad takes 2 or 3 inputs. Provided " + std::to_string(inputs.size()));

    Output<ngraph::Node> data = inputs.at(0);
    Output<ngraph::Node> pads_tensor = inputs.at(1);

    // Extract mode attribute (default to "constant" if not specified)
    const std::string mode = node.get_attribute_value<std::string>("mode", "constant");

    // Extract value attribute (default to 0.0 if not specified)
    const Output<ngraph::Node> value = (inputs.size() == 3) ? inputs.at(2) :
        default_opset::Constant::create(data.get_element_type(), Shape{}, {0.0});

    // Compute the number of dimensions
    const size_t num_dims = data.get_shape().size();

    // Compute the padding values
    std::vector<int64_t> pads;
    if (pads_tensor.get_shape().size() == 1) {
        pads = pads_tensor.get_shape_val();
    } else if (pads_tensor.get_shape().size() == 2) {
        pads = pads_tensor.get_shape_val()[0];
    } else {
        NGRAPH_CHECK(false, "Invalid shape for pads tensor. It should be 1D or 2D.");
    }

    NGRAPH_CHECK(pads.size() == 2 * num_dims,
                 "Invalid shape for pads tensor. It should have 2 * input_rank elements.");

    // Create the Pad operation
    std::shared_ptr<ngraph::op::Pad> pad_node;
    if (mode == "constant") {
        pad_node = std::make_shared<ngraph::op::Pad>(data, pads, value);
    } else if (mode == "reflect") {
        pad_node = std::make_shared<ngraph::op::Pad>(data, pads, ngraph::op::PadMode::REFLECT);
    } else if (mode == "edge") {
        pad_node = std::make_shared<ngraph::op::Pad>(data, pads, ngraph::op::PadMode::EDGE);
    } else {
        NGRAPH_CHECK(false, "Unsupported padding mode: " + mode);
    }

    return {pad_node};
}

}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
