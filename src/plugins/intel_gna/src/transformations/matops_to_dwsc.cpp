#include "matops_to_dwsc.hpp"

#include <memory>

using namespace ngraph;
using namespace op;

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/rt_info.hpp>

// Check if the previous node is convolution, groupconvolution or matmul followed by add 
static bool IsFusable(Output<Node>& parent) {
    if (nullptr == std::dynamic_pointer_cast<ngraph::opset8::GroupConvolution>(parent) ||
        nullptr == std::dynamic_pointer_cast<ngraph::opset8::Convolution>(parent) ||
        nullptr == std::dynamic_pointer_cast<ngraph::opset8::MatMul>(parent))
        return false;
    return true;
}

static bool Decompose(std::shared_ptr<ov::Node> math_node) {
    const Output<Node>& input = math_node->input_value(0);
    const Output<Node>& params = math_node->input_value(1);

    // extracting the dimensions
    auto input_shape = input.get_shape();
    auto params_shape = params.get_shape();
    auto auto_broadcast = math_node->get_autob();
    auto output_shape = math_node->get_output_shape(0);

    if (input_shape.size() == 0)
        return false;

    uint64_t N, C, H, W;
    switch (input_shape.size()) {
    case 4:                  // 3 dimensional case
        N = input_shape[0];  // batchsize
        C = input_shape[1];  // input channels (input depth)
        H = input_shape[2];  // input height
        W = input_shape[3];  // input widith
        break;
    case 2:  // 1 dimensional case (1 channel)
        N = 1;
        C = input_shape[1];
        H = 1;
        W = input_shape[0];
        break;
    default:
        return false;
    }

    uint64_t N_params, C_params, H_params, W_params;
    switch (params_shape.size()) {
    case 4:
        N_params = params_shape[0];
        C_params = params_shape[1];
        H_params = input_shape[2];
        W_params = input_shape[3];
        break;
    case 2:
        N_params = params_shape[0];
        C_params = params_shape[1];
        H_params = 1;
        W_params = 1;
        break;
    case 1:
        N_params = 1;
        C_params = params_shape[0];
        H_params = 1;
        W_params = 1;
        break;
    default:
        return false;
    }

    // getting the parent(previous) node and set of children nodes
    Output<Node>& parent = math_node->input_value(0);
    if (nullptr != std::dynamic_pointer_cast<ngraph::opset1::Multiply>(math_node) && IsFusable(parent))
        return false;

    auto G = C;             // number of groups (subdivisions of the input tensor)
    auto Co = (uint64_t)1;  // no output channels for one group of kernels (no kernels in a group)
    auto Ci = (uint64_t)1;  // no input channels in a group
    auto Kh = (uint64_t)1;  // kernel height
    auto Kw = (uint64_t)1;  // kernel width

    // Initialize dwsc (group convolution) kernel weights as 1s
    std::vector<float> dwsc_weights(Kh * Kw * G, 0.0f);
    float* dwsc_weight_ptr = dwsc_weights.data();
    if (nullptr == std::dynamic_pointer_cast<ngraph::opset1::Multiply>(math_node)) {
        for (uint32_t i = 0; i < Kh * Kw * G; i++)
            *(dwsc_weight_ptr + i) = 1;
    } else {
        auto weights_const =
            std::dynamic_pointer_cast<opset1::Constant>(math_node->input_value(1).get_node_shared_ptr());
        const float* weights_ptr = weights_const->get_data_ptr<float>();
        for (uint32_t i = 0; i < Kh * Kw * G; i++)
            *(dwsc_weight_ptr + i) = *(weights_ptr + i);
    }

    // Create a constant vector of weights
    auto dwsc_weights_const = op::Constant::create(ngraph::element::f32, Shape{G, Co, Ci, Kh, Kw}, dwsc_weights);
    dwsc_weights_const->set_friendly_name("dwsc_weights");

    // Initializing group convolution parameters
    const Strides& strides = Strides({1, 1});
    const CoordinateDiff& pads_begin = CoordinateDiff({0, 0});
    const CoordinateDiff& pads_end = CoordinateDiff({0, 0});
    const Strides& dilations = Strides({1, 1});
    const PadType& auto_pad = PadType::EXPLICIT;  // Explicit padding so we just use the zeros from above

    // Initializing the dwsc node
    std::shared_ptr<ov::op::v1::GroupConvolution> new_dwsc;
    if (4 != input_shape.size()) {
        std::shared_ptr<ov::op::v1::Reshape> input_4d;
        if (W > 1) {
            auto n_elements = input_shape[0] * input_shape[1];
            if (n_elements < 8 || n_elements > 65528 || n_elements % 8 != 0)
                return false;
            auto new_transpose =
                std::make_shared<op::Transpose>(math_node->input_value(0),
                                                op::Constant::create(element::Type_t::i64, Shape{2}, {1, 0}));
            input_4d = std::make_shared<ngraph::opset1::Reshape>(
                new_transpose->output(0),
                op::Constant::create(ngraph::element::i64, Shape{4}, {N, C, H, W})->output(0),
                false);
        } else
            input_4d = std::make_shared<ngraph::opset1::Reshape>(
                math_node->input_value(0),
                op::Constant::create(ngraph::element::i64, Shape{4}, {N, C, H, W})->output(0),
                false);
        new_dwsc = std::make_shared<opset1::GroupConvolution>(input_4d->output(0),
                                                              dwsc_weights_const->output(0),
                                                              strides,
                                                              pads_begin,
                                                              pads_end,
                                                              dilations,
                                                              auto_pad);
    } else {
        new_dwsc = std::make_shared<opset1::GroupConvolution>(math_node->input_value(0),
                                                              dwsc_weights_const->output(0),
                                                              strides,
                                                              pads_begin,
                                                              pads_end,
                                                              dilations,
                                                              auto_pad);
    }

    new_dwsc->set_friendly_name("replace_math_operation");

    std::shared_ptr<ov::Node> skip_node;
    if (nullptr == std::dynamic_pointer_cast<ngraph::opset1::Multiply>(math_node)) {
        // creating a bias node
        auto bias_const = std::dynamic_pointer_cast<opset1::Constant>(math_node->input_value(1).get_node_shared_ptr());
        const float* bias_ptr = bias_const->get_data_ptr<float>();
        std::vector<float> new_bias(N_params * C_params * H_params * W_params, 0.0f);
        float* new_bias_ptr = new_bias.data();

        if (nullptr == std::dynamic_pointer_cast<ngraph::opset1::Subtract>(math_node)) {
            for (size_t i = 0; i < N_params * C_params * H_params * W_params; i++)
                *(new_bias_ptr + i) = *(bias_ptr + i);
        } else {
            for (size_t i = 0; i < N_params * C_params * H_params * W_params; i++)
                *(new_bias_ptr + i) = -1 * (*(bias_ptr + i));
        }
        auto new_bias_const =
            op::Constant::create(ngraph::element::f32, Shape{N_params, C_params, H_params, W_params}, new_bias);

        // creating a new add node
        auto new_add = std::make_shared<opset1::Add>(new_dwsc->output(0), new_bias_const->output(0), auto_broadcast);
        skip_node = new_add;
    } else {
        skip_node = new_dwsc;
    }

    // Reshape for different input dimensions
    if (4 == input_shape.size()) {
        ngraph::replace_node(math_node, skip_node);
    } else {
        std::shared_ptr<ngraph::opset1::Reshape> new_reshape;
        if (W > 1) {
            new_reshape = std::make_shared<ngraph::opset1::Reshape>(
                skip_node->output(0),
                op::Constant::create(ngraph::element::i64, Shape{input_shape.size()}, {input_shape[1], input_shape[0]})
                    ->output(0),
                false);
            auto new_shape = new_reshape->output(0).get_shape();
            auto n_elements = new_shape[0] * new_shape[1];
            if (n_elements < 8 || n_elements > 65528 || n_elements % 8 != 0)
                return false;

            auto untranspose =
                std::make_shared<op::Transpose>(new_reshape->output(0),
                                                op::Constant::create(element::Type_t::i64, Shape{2}, {1, 0}));
            ngraph::replace_node(math_node, untranspose);
        } else {
            new_reshape = std::make_shared<ngraph::opset1::Reshape>(
                skip_node->output(0),
                op::Constant::create(ngraph::element::i64, Shape{input_shape.size()}, input_shape)->output(0),
                false);
            ngraph::replace_node(math_node, new_reshape);
        }
    }

    return true;
}

NGRAPH_RTTI_DEFINITION(ngraph::pass::AddDecomposition, "AddDecomposition", 0);
bool ngraph::pass::AddDecomposition::run_on_function(std::shared_ptr<ngraph::Function> f) {
    bool is_graph_modfied = false;
    for (auto& node : f->get_ordered_ops()) {
        auto addition = std::dynamic_pointer_cast<ngraph::opset1::Add>(node);
        if (nullptr == addition)
            continue;

        if (Decompose(addition))
            is_graph_modfied = true;
    }
    return is_graph_modfied;
}

NGRAPH_RTTI_DEFINITION(ngraph::pass::SubDecomposition, "SubDecomposition", 0);
bool ngraph::pass::SubDecomposition::run_on_function(std::shared_ptr<ngraph::Function> f) {
    bool is_graph_modfied = false;
    for (auto& node : f->get_ordered_ops()) {
        auto subtraction = std::dynamic_pointer_cast<ngraph::opset1::Subtract>(node);
        if (nullptr == subtraction)
            continue;

        if (Decompose(subtraction))
            is_graph_modfied = true;
    }

    return is_graph_modfied;
}

NGRAPH_RTTI_DEFINITION(ngraph::pass::MulDecomposition, "MulDecomposition", 0);
bool ngraph::pass::MulDecomposition::run_on_function(std::shared_ptr<ngraph::Function> f) {
    bool is_graph_modfied = false;
    for (auto& node : f->get_ordered_ops()) {
        auto multiplication = std::dynamic_pointer_cast<ngraph::opset1::Multiply>(node);
        if (nullptr == multiplication)
            continue;

        if (Decompose(multiplication))
            is_graph_modfied = true;
    }

    return is_graph_modfied;
}