#include <openvino/cc/ngraph/itt.hpp>

#include "transformations/reshape1dconvolution.hpp"

#include "transformations/utils/transformation_helper.hpp"

#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <transformations/utils/utils.hpp>
#include <ops/gna_convolution.hpp>
#include <ngraph/pass/manager.hpp>

#include <ops/gna_convolution.hpp> // TODO: remove it after debug

using namespace GNAPluginNS;

NGRAPH_RTTI_DEFINITION(GNAPluginNS::Reshape1DConvolution, "Reshape1DConvolution", 0);

namespace {

template <class T>
std::shared_ptr<ngraph::Node> convert(const ngraph::Output<ngraph::Node> & data, std::shared_ptr<T> node, ngraph::NodeVector & new_ops);

template <>
std::shared_ptr<ngraph::Node> convert(const ngraph::Output<ngraph::Node> & data, std::shared_ptr<ngraph::opset8::Convolution/*GNAPluginNS::Op::GNAConvolution*/> node, ngraph::NodeVector & new_ops) {
    // Update Convolution attributes with additional dimension
    auto new_strides = node->get_strides();
    auto new_dilations = node->get_dilations();
    auto new_pads_begin = node->get_pads_begin();
    auto new_pad_end = node->get_pads_end();

    new_strides.insert(new_strides.begin(), 1);
    new_dilations.insert(new_dilations.begin(), 1);
    new_pads_begin.insert(new_pads_begin.begin(), 0);
    new_pad_end.insert(new_pad_end.begin(), 0);

    ngraph::Shape new_weights_shape(node->input_value(1).get_shape());
    new_weights_shape.insert(new_weights_shape.begin() + 2, 1);
    auto weights = ngraph::op::util::reshapeTo(node->input_value(1), new_weights_shape);
    new_ops.push_back(weights);

    return std::make_shared<ngraph::opset8::Convolution/*GNAPluginNS::Op::GNAConvolution*/>(data,
                                                   weights,
                                                   new_strides,
                                                   new_pads_begin,
                                                   new_pad_end,
                                                   new_dilations,
                                                   node->get_auto_pad());
}

ngraph::matcher_pass_callback get_callback() {
    return [](ngraph::pattern::Matcher& m) {
        auto node = m.get_match_root();
        if (node->input(0).get_partial_shape().rank().get_length() != 3) {
            return false;
        }

        // Insert H dimension equal to 1
        auto input_shape = node->input(0).get_shape();
        auto output_shape = node->output(0).get_shape();

        input_shape.insert(input_shape.begin() + 2, 1);

        ngraph::NodeVector new_ops;

        // Reshape(input_shape)->Op->Reshape(output_shape)
        ngraph::Output<ngraph::Node> last = ngraph::op::util::reshapeTo(node->input_value(0), input_shape);
        last.get_node_shared_ptr()->set_friendly_name(node->get_friendly_name() + "/reshape_begin");
        new_ops.push_back(last.get_node_shared_ptr());

        if (auto conv = std::dynamic_pointer_cast<ngraph::opset8::Convolution/*GNAPluginNS::Op::GNAConvolution*/>(node))
            last = convert(last, conv, new_ops);

        last.get_node_shared_ptr()->set_friendly_name(node->get_friendly_name() + "/new");
        new_ops.push_back(last.get_node_shared_ptr());

        last = ngraph::op::util::reshapeTo(last, output_shape);
        last.get_node_shared_ptr()->set_friendly_name(node->get_friendly_name());
        new_ops.push_back(last.get_node_shared_ptr());

        ngraph::copy_runtime_info(node, new_ops);
        node->output(0).replace(last);
        return true;
    };
}

} // namespace

Reshape1DConvolution::Reshape1DConvolution() {
    MATCHER_SCOPE(Reshape1DConvolution);

    auto conv = ngraph::pattern::wrap_type<ngraph::opset8::Convolution/*GNAPluginNS::Op::GNAConvolution*/>(ngraph::pattern::has_static_shape());
    auto m = std::make_shared<ngraph::pattern::Matcher>(conv, "Reshape1DConvolution");
    this->register_matcher(m, get_callback());
}
