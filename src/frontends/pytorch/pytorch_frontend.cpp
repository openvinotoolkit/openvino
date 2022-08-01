#include <map>
#include <exception>
#include <memory>

#include <ngraph/opsets/opset7.hpp>
#include <ngraph/opsets/opset8.hpp>
#include "openvino/op/util/framework_node.hpp"

#include "pytorch_frontend.h"


namespace torch {
namespace jit {
namespace fuser {
namespace openvino {


#define OV_FRONTEND_REQUIRE(X) \
    do if(!(X)) { \
        throw std::runtime_error(std::string("[ ERROR ] Failed: ") + #X); \
    } while(false)

ngraph::element::Type to_ngraph_data_type(at::ScalarType dt) {
    switch (dt) {
        case at::ScalarType::Float:
            return ngraph::element::f32;
        case at::ScalarType::BFloat16:
            return ngraph::element::bf16;
        case at::kInt:
            return ngraph::element::i32;
        case at::kLong:
            return ngraph::element::i64;
        case at::ScalarType::QInt8:
            return ngraph::element::i8;   // TODO: Is it correct?
        case at::ScalarType::QUInt8:
            return ngraph::element::u8;   // TODO: Is it correct?
        default:
            TORCH_CHECK(false, "Not support data type ", dt);
    }
}

ngraph::PartialShape get_ov_shape(const Value* v) {
    std::vector<ngraph::Dimension> dims;
    if (v->type()->isSubtypeOf(TensorType::get())) {
        const auto &tt = v->type()->cast<TensorType>();
        //std::cerr << "sizes = " << tt->sizes() << "\n";
        //std::cerr << "dims = " << tt->dim() << "\n";

        const auto &sizes = tt->sizes();
        if (sizes.sizes())
            for (auto &d : *sizes.sizes())
                dims.push_back(d.value_or(-1));
    } else {
        auto value = toIValue(v);

        if (value->isIntList()) {
            auto vec = value->toIntVector();
            dims = {vec.size()};
        } else if (value->isInt()) {
            auto val = value->toInt();
            dims = {};
        } else if (value->isDouble()) {
            auto val = value->toDouble();
            dims = {};
        } else {
            //std::cerr << "[ ERROR ] Cannot retrieve shape for not a tensor value\n";
            return ngraph::PartialShape::dynamic();
        }
    }
    return ngraph::PartialShape(dims);
}

ngraph::element::Type get_ov_element_type(const Value* v) {
    if (v->type()->isSubtypeOf(TensorType::get())) {
        const auto& tt = v->type()->cast<TensorType>();

        if (tt->scalarType().has_value()) {
            return to_ngraph_data_type(tt->scalarType().value());
        } else {
            return ngraph::element::f32;
        }
    } else {
        //std::cerr << "[ ERROR ] Cannot retrieve type for not a tensor value\n";
        return ngraph::element::dynamic;
    }
}

std::vector<int32_t> get_transpose_order(const Value* v) {
    std::vector<int32_t> idx;
    if (v->type()->isSubtypeOf(TensorType::get())) {
        ngraph::PartialShape dims;
        const auto& tt = v->type()->cast<TensorType>();

        const auto& strides = tt->strides();
        if (strides.sizes())
            for (auto& d : *strides.sizes()) {
                dims.push_back(d.value_or(-1));
            }
        if (dims.is_dynamic()) {
            throw std::runtime_error("Cannot retrieve transpose order for dynamic strides");
        }
        const auto& s = dims.get_shape();
        idx.resize(s.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::stable_sort(idx.begin(), idx.end(), [&s](size_t lhs, size_t rhs) {return s[lhs] > s[rhs];});
    } else {
        //std::cerr << "[ ERROR ] Cannot retrieve strides for not a tensor value\n";
        return {};
    }
    return idx;
}

void mark_node (std::shared_ptr<ngraph::Node> ov_node, const Node* pt_node) {
    auto& rt_info = ov_node->get_rt_info();
    auto rt_info_pt = rt_info.find("pt_node");
    if(rt_info_pt == rt_info.end()) {
        rt_info_pt = rt_info.insert(std::map<std::string, ov::Any>::value_type("pt_node", std::set<const Node*>())).first;
    }
    rt_info_pt->second.as<std::set<const Node*>>().insert(pt_node);
}

typedef std::map<size_t, ngraph::Output<ngraph::Node>> TensorMap;

class NodeDecoder : public std::enable_shared_from_this<NodeDecoder> {
public:

    NodeDecoder (std::shared_ptr<Graph> _graph, const Node* _node, const TensorMap& _tensor_map, const TensorArgs& _graph_tensors) :
        graph(_graph), node(_node), tensor_map(_tensor_map), graph_tensors(_graph_tensors) {}

    // Do not search for input in tensor map; try to access it as a constant of specified type T and return its value
    template <typename T>
    T const_input (size_t index) const;

    // FIXME: This method is provided to support a work-around to prove some perf expectations. It should be deleted.
    // Extract tensor value from one of the torch graph inputs and represent it as constant node
    ngraph::Output<ngraph::Node> const_node_input (size_t index) {
        OV_FRONTEND_REQUIRE(!input_is_none(index));
        auto id = node->inputs()[index]->unique();
        // search for id in graph inputs
        // TODO: Search for a better way
        for(size_t i = 0; i < graph->inputs().size(); ++i) {
            if(graph->inputs()[i]->unique() == id) {
                if(graph_tensors.empty()) {
                    throw std::runtime_error("const_node_input failed: there are no real input tensors");
                }
                return make_constant(graph_tensors[i]);
            }
        }

        throw std::runtime_error("const_node_input failed: cannot find input with a given id among graph inputs");
    }

    // Search for input in tensor map and return an output port for already converted op
    ngraph::Output<ngraph::Node> input (size_t index) const {
        //std::cerr << "Trying to map input to ngraph...";
        OV_FRONTEND_REQUIRE(!input_is_none(index));
        return tensor_map.at(node->inputs()[index]->unique());
    }

    ngraph::OutputVector inputs () const {
        ngraph::OutputVector res;
        for (auto& input : node->inputs()) {
            //std::cerr << "Searching for input: " << input->unique() << "\n";
            OV_FRONTEND_REQUIRE(tensor_map.find(input->unique()) != tensor_map.end());
            res.push_back(tensor_map.at(input->unique()));
        }
        return res;
    }

    bool input_is_none (size_t index) const {
        return node->inputs().size() <= index || !node->inputs()[index]->mustNotBeNone();
    }

    // Convert the resulting value of this node to ngraph Constant; works correctly only for nodes that produce
    // constant value, naturally for prim::Constant
    ngraph::OutputVector as_constant ();

    torch::jit::NodeKind get_op_type() const {
        return node->kind();
    }

    size_t num_of_outputs () const {
        return node->outputs().size();
    }

    c10::ArrayRef<const torch::jit::Value*> outputs () const {
        return node->outputs();
    }

    std::shared_ptr<ngraph::Node> mark_node (std::shared_ptr<ngraph::Node> ov_node) const {
        openvino::mark_node(ov_node, node);
        return ov_node;
    }

    void mark_nodes (std::vector<std::shared_ptr<ngraph::Node>> ov_nodes) const {
        for (auto& ov_node : ov_nodes) {
            openvino::mark_node(ov_node, node);
        }
    }

    ngraph::Output<ngraph::Node> mark_output (ngraph::Output<ngraph::Node> ov_output) const {
        openvino::mark_node(ov_output.get_node_shared_ptr(), node);
        return ov_output;
    }

private:

    ngraph::Output<ngraph::Node> make_constant(const at::Tensor& tensor) const;

    std::shared_ptr<Graph> graph;
    const Node* node;
    const TensorMap& tensor_map;
    const TensorArgs& graph_tensors;
};

template <typename Tdst, typename Tsrc>
std::vector<Tdst> convert_vector (const std::vector<Tsrc>& in) {
    return std::vector<Tdst>(in.begin(), in.end());
}

template <>
std::vector<int64_t> NodeDecoder::const_input<std::vector<int64_t>> (size_t index) const {
    OV_FRONTEND_REQUIRE(!input_is_none(index));
    return toIValue(node->input(index))->toIntVector();
}

template <>
std::string NodeDecoder::const_input<std::string> (size_t index) const {
    OV_FRONTEND_REQUIRE(!input_is_none(index));
    return toIValue(node->input(index))->toString()->string();
}

template <>
ngraph::Strides NodeDecoder::const_input<ngraph::Strides> (size_t index) const {
    return convert_vector<ngraph::Strides::value_type>(const_input<std::vector<int64_t>>(index));
}

template <>
ngraph::CoordinateDiff NodeDecoder::const_input<ngraph::CoordinateDiff> (size_t index) const {
    return convert_vector<ngraph::CoordinateDiff::value_type>(const_input<std::vector<int64_t>>(index));
}

template <>
ngraph::Shape NodeDecoder::const_input<ngraph::Shape> (size_t index) const {
    return convert_vector<ngraph::Shape::value_type>(const_input<std::vector<int64_t>>(index));
}

template <>
int64_t NodeDecoder::const_input<int64_t> (size_t index) const {
    OV_FRONTEND_REQUIRE(!input_is_none(index));
    return toIValue(node->input(index))->toInt();
}

template <>
bool NodeDecoder::const_input<bool> (size_t index) const {
    OV_FRONTEND_REQUIRE(!input_is_none(index));
    return toIValue(node->input(index))->toBool();
}

template <>
double NodeDecoder::const_input<double> (size_t index) const {
    OV_FRONTEND_REQUIRE(!input_is_none(index));
    return toIValue(node->input(index))->toDouble();
}

template <>
float NodeDecoder::const_input<float> (size_t index) const {
    return const_input<double>(index);
}

ngraph::Output<ngraph::Node> NodeDecoder::make_constant(const at::Tensor& tensor) const {
    auto shape = convert_vector<ngraph::Shape::value_type>(tensor.sizes().vec());
    // TODO: Check strides; now, for our first experiments we expect that they are trivial and don't introduce gaps
    //auto strides = t.strides().vec();
    auto type = to_ngraph_data_type(tensor.scalar_type());
    return std::make_shared<ngraph::opset7::Constant>(type, shape, tensor.data_ptr());
}


class PtFrameworkNode : public ov::op::util::FrameworkNode {
public:
    OPENVINO_OP("PtFrameworkNode", "util", ::ov::op::util::FrameworkNode);

    PtFrameworkNode(const std::shared_ptr<NodeDecoder>& decoder, const ngraph::OutputVector& inputs)
        : ov::op::util::FrameworkNode(inputs, decoder->num_of_outputs()),
          m_decoder(decoder) {
        ov::op::util::FrameworkNodeAttrs attrs;
        //std::cerr << "[ DEBUG ] Making  PtFrameworkNode for " << m_decoder->get_op_type().toQualString() << "\n";
        attrs.set_type_name("PTFrameworkNode");
        attrs["PtTypeName"] = m_decoder->get_op_type().toQualString();
        set_attrs(attrs);

        // Set output shapes and types if recognized

        auto outputs = decoder->outputs();
        for(size_t i = 0; i < outputs.size(); ++i) {
            ngraph::PartialShape ps;
            ngraph::element::Type et = ngraph::element::dynamic;
            // FIXME: ROUGH
            try {
                ps = get_ov_shape(outputs[i]);
            } catch (...) {
                // nothing, means the info cannot be queried and remains unknown
            }
            // FIXME: ROUGH
            try {
                et = get_ov_element_type(outputs[i]);
            } catch (...) {
                // nothing, means the info cannot be queried and remains unknown
                //std::cerr << "[ ERROR ] Cannot retrieve type\n";
            }
            set_output_type(i, et, ps);
        }
    }

    std::shared_ptr<Node> clone_with_new_inputs(const ngraph::OutputVector& inputs) const override {
        return std::make_shared<PtFrameworkNode>(m_decoder, inputs);
    }

    torch::jit::NodeKind get_op_type() const {
        return m_decoder->get_op_type();
    }

    NodeDecoder* get_decoder() const {
        return m_decoder.get();
    }

private:
    std::shared_ptr<NodeDecoder> m_decoder;
};


ngraph::OutputVector NodeDecoder::as_constant () {
    auto value = toIValue(node->output(0));
    std::shared_ptr<ngraph::Node> constant;
    if (value->type()->isSubtypeOf(TensorType::get())) {
        // TODO: Too long indirect path; can be shorter if we hold value pointer as a field of the class
        constant = make_constant(value->toTensor()).get_node_shared_ptr();
    } else if (value->isIntList()) {
        auto vec = value->toIntVector();
        constant = ngraph::opset7::Constant::create(ov::element::i64, {vec.size()}, vec);
    } else if (value->isInt()) {
        auto val = value->toInt();
        constant = ngraph::opset7::Constant::create(ov::element::i64, {}, {val});
    } else if (value->isDouble()) {
        auto val = value->toDouble();
        constant = ngraph::opset7::Constant::create(ov::element::f32, {}, {val});
    } else {
        //std::cerr << "[ ERROR ] Constant type " << value->tagKind() << " is not recognized; replaced by PT FrameworkNode\n";
        constant = std::make_shared<PtFrameworkNode>(shared_from_this(), ngraph::OutputVector{});
    }
    return { mark_node(constant) };
}


ngraph::Output<ngraph::Node> make_optional_bias(
        ngraph::Output<ngraph::Node> base_op,
        const std::shared_ptr<NodeDecoder>& context,
        size_t bias_input_idx,
        std::vector<int> unsqueeze_dims = {})
{
    using namespace ngraph;
    using std::make_shared;

    if(!context->input_is_none(bias_input_idx)) {
        auto bias = context->input(bias_input_idx);
        if(!unsqueeze_dims.empty()) {
            auto indices = opset7::Constant::create(element::i32, { unsqueeze_dims.size() }, unsqueeze_dims);
            context->mark_node(indices);
            bias = make_shared<opset7::Unsqueeze>(bias, indices);
            context->mark_output(bias);
        }
        return make_shared<opset7::Add>(context->mark_output(base_op), bias);
    } else {
        return base_op;
    }
}

std::shared_ptr<ov::Node> get_rank_node(ov::Output<ov::Node> node) {
    auto shape = std::make_shared<ngraph::opset8::ShapeOf>(node);
    return std::make_shared<ngraph::opset8::ShapeOf>(shape);
}

ngraph::Output<ngraph::Node> reshape_kernel_for_group(
        std::shared_ptr<NodeDecoder> context,
        ngraph::Output<ngraph::Node> input,
        ngraph::Output<ngraph::Node> kernel,
        int64_t groups)
{
    using namespace ngraph;
    using std::make_shared;

    auto in_shape = std::make_shared<ngraph::opset8::ShapeOf>(input);
    auto c_in_idx = opset8::Constant::create(element::i64, Shape{}, {1});
    auto axis_0 = opset8::Constant::create(element::i64, Shape{}, {0});
    auto in_shape_1 = make_shared<opset8::Gather>(in_shape, c_in_idx, axis_0);
    auto in_shape_1_uns = make_shared<opset8::Unsqueeze>(in_shape_1, axis_0);
    auto groups_const = opset8::Constant::create(element::i64, Shape{1}, {groups});
    auto c_in_value = make_shared<opset8::Divide>(in_shape_1_uns, groups_const);

    auto kernel_shape = std::make_shared<ngraph::opset8::ShapeOf>(kernel);
    auto c_out_idx = opset8::Constant::create(element::i64, Shape{}, {0});
    auto kernel_shape_0 = make_shared<opset8::Gather>(kernel_shape, c_out_idx, axis_0);
    auto kernel_shape_0_uns = make_shared<opset8::Unsqueeze>(kernel_shape_0, axis_0);
    auto c_out_value = make_shared<opset8::Divide>(kernel_shape_0_uns, groups_const);

    auto start = opset8::Constant::create(element::i64, Shape{1}, {2});
    auto stop = opset8::Constant::create(element::i64, Shape{1}, {INT_MAX});
    auto step = opset8::Constant::create(element::i64, Shape{1}, {1});
    auto remaining_shape = make_shared<opset8::Slice>(kernel_shape, start, stop, step);

    auto new_kernel_shape = make_shared<opset8::Concat>(OutputVector{groups_const, c_out_value, c_in_value, remaining_shape}, 0);
    context->mark_nodes({in_shape, c_in_idx, axis_0, in_shape_1, in_shape_1_uns, groups_const, c_in_value,
                        kernel_shape, c_out_idx, kernel_shape_0, kernel_shape_0_uns, c_out_value, start, stop, step, remaining_shape, new_kernel_shape});
    return make_shared<opset8::Reshape>(kernel, new_kernel_shape, false);
}

ngraph::OutputVector convert_node(std::shared_ptr<Graph> graph, const Node* node, const TensorMap& tensor_map, const TensorArgs& graph_tensors) {
    using ints = std::vector<int64_t>;
    using namespace ngraph;
    using std::make_shared;

    //std::cerr << "---\nAttempting to convert " << node->kind().toQualString() << "\n";
    //node->dump();

    auto context = make_shared<NodeDecoder>(graph, node, tensor_map, graph_tensors);

    try {
    switch (node->kind()) {

        case aten::relu: {
            return { context->mark_node(make_shared<opset7::Relu>(context->input(0))) };
        }

        case aten::conv2d: {

            auto strides = context->const_input<Strides>(3);
            auto pads_begin = context->const_input<CoordinateDiff>(4);  // FIXME: The same 4 is used twice
            auto pads_end = context->const_input<CoordinateDiff>(4);    // FIXME: The same 4 is used twice
            auto dilations = context->const_input<Strides>(5);
            auto groups = context->const_input<int64_t>(6);

            std::shared_ptr<ov::Node> conv;
            if (groups == 1) {
                conv = make_shared<opset7::Convolution>(
                    context->input(0),
                    context->input(1),
                    strides,
                    pads_begin,
                    pads_end,
                    dilations
                );
            } else {
                conv = make_shared<opset7::GroupConvolution>(
                    context->input(0),
                    reshape_kernel_for_group(context, context->input(0), context->input(1), groups),
                    strides,
                    pads_begin,
                    pads_end,
                    dilations
                );
            }

            // FIXME: Doesn't work for dynamic rank
            // FIXME: Works for 2D convolutions only
            return { context->mark_output(make_optional_bias(conv, context, 2, {-2, -1})) };
        }

        case aten::_convolution: {
            bool transposed = context->const_input<bool>(6);
            // TODO: Handle this temporary limitation
            OV_FRONTEND_REQUIRE(!transposed);

            auto strides = context->const_input<Strides>(3);
            auto pads_begin = context->const_input<CoordinateDiff>(4);  // FIXME: The same 4 is used twice
            auto pads_end = context->const_input<CoordinateDiff>(4);    // FIXME: The same 4 is used twice
            auto dilations = context->const_input<Strides>(5);
            // TODO: Handle skipped input 7 (6 was used above) -- what is it for?
            auto groups = context->const_input<int64_t>(8);

            std::shared_ptr<ov::Node> conv;
            if (groups == 1) {
                conv = make_shared<opset7::Convolution>(
                    context->input(0),
                    context->input(1),
                    strides,
                    pads_begin,
                    pads_end,
                    dilations
                );
            } else {
                conv = make_shared<opset7::GroupConvolution>(
                    context->input(0),
                    context->mark_output(reshape_kernel_for_group(context, context->input(0), context->input(1), groups)),
                    strides,
                    pads_begin,
                    pads_end,
                    dilations
                );
            }

            // FIXME: Doesn't work for dynamic rank
            // FIXME: Works for 2D convolutions only
            return { context->mark_output(make_optional_bias(conv, context, 2, {-2, -1})) };
        }

        case aten::batch_norm: {
            auto training = context->const_input<bool>(5);
            OV_FRONTEND_REQUIRE(!training);   // TODO: support bn training
            return { context->mark_node(make_shared<opset7::BatchNormInference>(
                        context->input(0),
                        context->input(1),
                        context->input(2),
                        context->input(3),
                        context->input(4),
                        context->const_input<float>(7)  // epsilon
                    )) };
        }

        case aten::layer_norm: {
            auto normalized_shape = context->const_input<Shape>(1);
            auto in_pshape_last_dim = *context->input(0).get_partial_shape().rbegin();
            OV_FRONTEND_REQUIRE(
                normalized_shape.size() == 1 && in_pshape_last_dim.is_static() &&
                static_cast<uint64_t>(in_pshape_last_dim.get_length()) == normalized_shape.back());
            auto eps = context->const_input<float>(4);
            auto axes = context->mark_node(opset7::Constant::create(element::i64, Shape{1}, {-1})); // TODO: support any dimention
            auto mvn = context->mark_node(make_shared<opset7::MVN>(context->input(0), axes, true, eps, op::MVNEpsMode::INSIDE_SQRT));
            std::shared_ptr<ov::Node> out_node = std::dynamic_pointer_cast<ov::Node>(mvn);
            if (!context->input_is_none(2)) {
                auto mul = make_shared<opset7::Multiply>(out_node, context->input(2));
                out_node = std::dynamic_pointer_cast<ov::Node>(mul);
            }
            if (!context->input_is_none(3)) {
                auto add = make_shared<opset7::Add>(out_node, context->input(3));
                out_node = std::dynamic_pointer_cast<ov::Node>(add);
            }
            return { context->mark_node(out_node) };
        }

        case aten::add: {
            return { context->mark_node(make_shared<opset7::Add>(context->input(0), context->input(1))) };
        }

        case aten::mul: {
            return { context->mark_node(make_shared<opset7::Multiply>(context->input(0), context->input(1))) };
        }

        case aten::div: {
            auto pythondiv = false;
            if (!context->input_is_none(2)) {
                auto rounding_mode = context->const_input<std::string>(2);
                if (rounding_mode == "floor") {
                    pythondiv = true;
                } else if (rounding_mode == "trunc") {
                    pythondiv = true;
                    //break;
                }
            }
            return { context->mark_node(make_shared<opset7::Divide>(context->input(0), context->input(1), pythondiv)) };
        }

        case aten::tanh: {
            return { context->mark_node(make_shared<opset7::Tanh>(context->input(0))) };
        }

        case aten::elu: {
            auto alpha = context->const_input<float>(1);
            return { context->mark_node(make_shared<opset7::Elu>(context->input(0), alpha)) };
        }

        case aten::sigmoid: {
            return { context->mark_node(make_shared<opset7::Sigmoid>(context->input(0))) };
        }

        case aten::gelu: {
            return { context->mark_node(make_shared<opset7::Gelu>(context->input(0))) };
        }

        case aten::sqrt: {
            return { context->mark_node(make_shared<opset7::Sqrt>(context->input(0))) };
        }

        case aten::abs: {
            return { context->mark_node(make_shared<opset7::Abs>(context->input(0))) };
        }

        case aten::square:  {
            auto input_0 = context->input(0);
            auto const_2 = context->mark_node(opset7::Constant::create(input_0.get_element_type(), Shape{1}, {2}));
            return { context->mark_node(make_shared<opset7::Power>(input_0, const_2)) };
        }

        case aten::hardtanh: {
            auto min = context->const_input<float>(1);
            auto max = context->const_input<float>(2);
            return { context->mark_node(make_shared<opset7::Clamp>(context->input(0), min, max)) };
        }

        case aten::hardsigmoid: {
            return { context->mark_node(make_shared<opset7::HSigmoid>(context->input(0))) };
        }

        case aten::hardswish: {
            return { context->mark_node(make_shared<opset7::HSwish>(context->input(0))) };
        }

        case aten::relu6: {
            return { context->mark_node(make_shared<opset7::Clamp>(context->input(0), 0., 6.)) };
        }

        case aten::softmax: {
            auto axis = context->const_input<int64_t>(1);
            if (axis < 0) {
                auto in_rank = context->input(0).get_partial_shape().rank();
                OV_FRONTEND_REQUIRE(in_rank.is_static());
                axis = in_rank.get_length() + axis;
            }
            return { context->mark_node(make_shared<opset7::Softmax>(context->input(0), static_cast<size_t>(axis))) };
        }

        case aten::cat: {
          // aten::cat needs a special handling since it takes a Tensor[] as
          // input. We set the inputs of ListConstruct as the inputs of cat.
          //
          // Pytorch IR:                              LLGA sees:
          //     %a    %b     %c          %dim              %a    %b    %c
          //      \     |     /             |                \     |    /
          //   prim::ListConstruct   prim::Constant     llga::Concat[axis=%dim]
          //                    \      /
          //                    aten::cat
          auto listConstruct = context->input(0).get_node();
          auto listConstruct_fw_node = dynamic_cast<PtFrameworkNode*>(listConstruct);
          OV_FRONTEND_REQUIRE(listConstruct_fw_node);
          OV_FRONTEND_REQUIRE(listConstruct_fw_node->get_decoder()->get_op_type() == prim::ListConstruct);
          auto axis = context->const_input<int64_t>(1);
          OutputVector inputs;
          for (auto& input : listConstruct->inputs()) {
            inputs.push_back(input.get_source_output());
          }
          auto result = context->mark_node(make_shared<opset7::Concat>(inputs, axis));
          auto list_set = listConstruct_fw_node->get_rt_info()["pt_node"].as<std::set<const Node*>>();
          result->get_rt_info()["pt_node"].as<std::set<const Node*>>().insert(list_set.begin(), list_set.end());
          return { result };
        }

        case aten::matmul: {
            return { context->mark_node(make_shared<opset7::MatMul>(context->input(0), context->input(1))) };
        }

        case aten::mm: {
            return { context->mark_node(make_shared<opset7::MatMul>(context->input(0), context->input(1))) };
        }

        case aten::linear: {
            auto matmul = make_shared<opset7::MatMul>(context->input(0), context->input(1), false, true);
            return { context->mark_output(make_optional_bias(matmul, context, 2)) };
        }

        case aten::max_pool2d: {
            auto kernel = context->const_input<Shape>(1);
            auto strides = context->const_input<Strides>(2);
            auto pads_begin = context->const_input<Shape>(3);  // FIXME: The same 3 is used twice
            auto pads_end = context->const_input<Shape>(3);    // FIXME: The same 3 is used twice
            auto dilations = context->const_input<Strides>(4);
            auto rounding_type = context->const_input<bool>(5) ? op::RoundingType::CEIL : op::RoundingType::FLOOR;

            // TODO: Upgrade to opset8::MaxPool to use dilations; for now we suppose they are all zeros
            return { context->mark_node(make_shared<opset7::MaxPool>(
                    context->input(0), strides, /*dilations,*/ pads_begin, pads_end, kernel, rounding_type)) };
        }

        case aten::avg_pool2d: {
            auto kernel = context->const_input<Shape>(1);
            auto strides = context->const_input<Strides>(2);
            auto pads_begin = context->const_input<Shape>(3);  // FIXME: The same 3 is used twice
            auto pads_end = context->const_input<Shape>(3);    // FIXME: The same 3 is used twice
            auto rounding_type = context->const_input<bool>(4) ? op::RoundingType::CEIL : op::RoundingType::FLOOR;
            auto exclude_pad = !context->const_input<bool>(5);
            // TODO: support divisor override
            // auto divisor_override = context->const_input<int64_t>(6);

            return { context->mark_node(make_shared<opset7::AvgPool>(
                    context->input(0), strides, pads_begin, pads_end, kernel, exclude_pad, rounding_type)) };
        }

        case aten::adaptive_avg_pool2d: {
            return { context->mark_node(make_shared<opset8::AdaptiveAvgPool>(context->input(0), context->input(1))) };
        }

        case aten::adaptive_max_pool2d: {
            auto adaptive_max_pool = context->mark_node(make_shared<opset8::AdaptiveMaxPool>(context->input(0), context->input(1)));
            auto return_indices = context->const_input<bool>(2);
            OutputVector res{adaptive_max_pool->output(0)};
            if (return_indices) {
                res.push_back(adaptive_max_pool->output(1));
            }
            return res;
        }

        case aten::mean: {
            auto keep_dims = context->const_input<bool>(2);            
            OV_FRONTEND_REQUIRE(context->input_is_none(3));
            return { context->mark_node(make_shared<opset8::ReduceMean>(context->input(0), context->input(1), keep_dims)) };
        }

        case aten::flatten: {
            auto start_dim = context->const_input<int64_t>(1);
            auto end_dim = context->const_input<int64_t>(2);
            auto data_pshape = context->input(0).get_partial_shape();
            OV_FRONTEND_REQUIRE(data_pshape.rank().is_static()); // TODO: support dynamic rank
            auto rank = data_pshape.rank().get_length();
            if (start_dim < 0) {
                start_dim = rank + start_dim;
            }
            if (end_dim < 0) {
                end_dim = rank + end_dim;
            }
            OV_FRONTEND_REQUIRE(start_dim < end_dim);
            auto delta = end_dim - start_dim;
            std::vector<int64_t> new_shape(rank - delta, 0);
            new_shape[start_dim] = -1;
            auto new_shape_const = context->mark_node(opset7::Constant::create(element::i64, {new_shape.size()}, new_shape));
            return { context->mark_node(make_shared<opset8::Reshape>(context->input(0), new_shape_const, true)) };
        }

        case prim::NumToTensor:
        case aten::contiguous: {
            // Do nothing
            return { context->mark_node(context->input(0).get_node_shared_ptr()) };
        }

        case aten::as_tensor: {
            OV_FRONTEND_REQUIRE(context->const_input<int64_t>(1) == 6);
            OV_FRONTEND_REQUIRE(context->input_is_none(2));
            //auto new_shape_const = context->mark_node(opset7::Constant::create(element::i64, {1}, {1}));
            //return { context->mark_node(std::make_shared<opset8::Reshape>(context->input(0), new_shape_const->output(0), true)) };
            return { context->mark_output(context->input(0)) };
        }

        case aten::Int: {
            return { context->mark_node(make_shared<opset8::Convert>(context->input(0), element::i64)) };
        }

        case aten::to: {
            auto dtype = element::f32;
            // TODO: figure out all inputs meaning
            OV_FRONTEND_REQUIRE(context->const_input<int64_t>(1) == 6);
            OV_FRONTEND_REQUIRE(context->const_input<bool>(2) == false);
            OV_FRONTEND_REQUIRE(context->const_input<bool>(3) == false);
            OV_FRONTEND_REQUIRE(context->input_is_none(4));
            return { context->mark_node(make_shared<opset8::Convert>(context->input(0), dtype)) };
        }

        case aten::permute: {
            return { context->mark_node(make_shared<opset7::Transpose>(context->input(0), context->input(1))) };
        }

        case aten::embedding: {
            // TODO: find out the meaning of input idx 2
            OV_FRONTEND_REQUIRE(context->const_input<bool>(3) == false);
            OV_FRONTEND_REQUIRE(context->const_input<bool>(4) == false);
            auto axis_0 = context->mark_node(opset8::Constant::create(element::i64, Shape{}, {0}));
            return { context->mark_node(make_shared<opset7::Gather>(context->input(0), context->input(1), axis_0)) };
        }

        case aten::transpose: {
            auto dim0 = context->const_input<int64_t>(1);
            auto dim1 = context->const_input<int64_t>(2);
            auto data_pshape = context->input(0).get_partial_shape();
            auto rank = data_pshape.rank();
            OV_FRONTEND_REQUIRE(rank.is_static());
            auto _rank = rank.get_length();
            if (dim0 < 0) {
                dim0 = _rank + dim0;
            }
            if (dim1 < 0) {
                dim1 = _rank + dim1;
            }
            OV_FRONTEND_REQUIRE(dim0 > 0 && dim1 > 0);
            OV_FRONTEND_REQUIRE(dim0 < _rank && dim1 < _rank);
            std::vector<int64_t> order(_rank, 0);
            std::iota(order.begin(), order.end(), 0);
            std::swap(order[dim0], order[dim1]);
            auto order_const = context->mark_node(opset7::Constant::create(element::i64, {order.size()}, order));
            return { context->mark_node(make_shared<opset7::Transpose>(context->input(0), order_const)) };
        }

        case aten::size: {
            OV_FRONTEND_REQUIRE(!context->input_is_none(1));
            auto shape = context->mark_node(make_shared<opset8::ShapeOf>(context->input(0)));
            auto axis_0 = context->mark_node(opset8::Constant::create(element::i64, Shape{}, {0}));
            return { context->mark_node(make_shared<opset8::Gather>(shape, context->input(1), axis_0)) };
        }

        case aten::view: {
            auto shape_node = context->input(1).get_node();
            auto shape_node_fw_node = dynamic_cast<PtFrameworkNode*>(shape_node);
            std::shared_ptr<ov::Node> reshape;
            if (shape_node_fw_node->get_decoder()->get_op_type() == prim::ListConstruct) {
                // TODO: maybe use pt shape instead of whole shape subgraph, because it may be more efficent
                OutputVector inputs;
                auto axis_0 = context->mark_node(opset8::Constant::create(element::i64, Shape{}, {0}));
                for (auto& input : shape_node->inputs()) {
                    auto rank = input.get_partial_shape().rank();
                    OV_FRONTEND_REQUIRE(rank.is_dynamic() || rank.get_length() == 0);
                    auto unsqueeze = context->mark_node(make_shared<opset7::Unsqueeze>(input.get_source_output(), axis_0));
                    inputs.push_back(unsqueeze);
                }
                auto concat = context->mark_node(make_shared<opset7::Concat>(inputs, 0));
                reshape = context->mark_node(make_shared<opset7::Reshape>(context->input(0), concat, false));
                auto list_set = shape_node_fw_node->get_rt_info()["pt_node"].as<std::set<const Node*>>();
                reshape->get_rt_info()["pt_node"].as<std::set<const Node*>>().insert(list_set.begin(), list_set.end());
            } else {
                reshape = context->mark_node(make_shared<opset7::Reshape>(context->input(0), context->input(1), false));
            }
            return { reshape };
        }

        case aten::unsqueeze: {
            return { context->mark_node(make_shared<opset8::Unsqueeze>(context->input(0), context->input(1))) };
        }

        case aten::rsub: {
            // reverse aten::sub other - self * alpha
            auto alpha_casted = context->mark_node(make_shared<opset8::Convert>(context->input(2), context->input(0).get_element_type()));
            auto alpha_mul = context->mark_node(make_shared<opset8::Multiply>(context->input(0), alpha_casted));
            return { context->mark_node(make_shared<opset8::Subtract>(context->input(1), alpha_mul)) };
        }

        case aten::slice: {
            ov::Output<ov::Node> dim = context->input(1);
            ov::Output<ov::Node> start = context->input(2);
            ov::Output<ov::Node> end = context->input(3);
            ov::Output<ov::Node> step = context->input(4);

            auto axis_0 = context->mark_node(opset8::Constant::create(element::i64, Shape{}, {0}));
            if (dim.get_partial_shape().rank().is_static() && dim.get_partial_shape().rank().get_length() == 0) {
                dim = context->mark_node(make_shared<opset8::Unsqueeze>(dim, axis_0));
            }
            if (start.get_partial_shape().rank().is_static() && start.get_partial_shape().rank().get_length() == 0) {
                start = context->mark_node(make_shared<opset8::Unsqueeze>(start, axis_0));
            }
            if (end.get_partial_shape().rank().is_static() && end.get_partial_shape().rank().get_length() == 0) {
                end = context->mark_node(make_shared<opset8::Unsqueeze>(end, axis_0));
            }
            if (step.get_partial_shape().rank().is_static() && step.get_partial_shape().rank().get_length() == 0) {
                step = context->mark_node(make_shared<opset8::Unsqueeze>(step, axis_0));
            }
            return { context->mark_node(make_shared<opset8::Slice>(context->input(0), start, end, step, dim)) };
        }

        case prim::ConstantChunk: {
            auto chunks = node->i(attr::chunks); // FIXME: create actual attribute function
            auto dim = node->i(attr::dim);
            auto dim_const = context->mark_node(opset8::Constant::create(element::i64, Shape{}, {dim}));
            auto split = context->mark_node(make_shared<opset8::Split>(context->input(0), dim_const, chunks));
            return split->outputs();
        }

        case prim::Constant: {
            return context->as_constant();
        }

    }
    }
    catch (...) {
        std::cout << "Some exception happened during convertion of node: " << *node << std::endl;
    }
    //if (node->kind() != prim::ListConstruct) {
    //    std::cout << "Making unsupported " << node->kind().toQualString() << std::endl;
    //    node->dump();
    //}
    return context->mark_node(make_shared<PtFrameworkNode>(context, context->inputs()))->outputs();

}

std::shared_ptr<ngraph::Function> convert(std::shared_ptr<Graph> graph, const TensorArgs& rt_inputs) {

    // Torch JIT Value ID to ngraph::Node map
    TensorMap tensor_map;

    ngraph::ParameterVector parameters;

    //graph->dump();

    // Go over all graph inputs and register them in the tensor map:
    for (auto* value: graph->inputs()) {
        ngraph::PartialShape ps = get_ov_shape(value);
        auto parameter = std::make_shared<ngraph::opset7::Parameter>(get_ov_element_type(value), ps);
        parameters.push_back(parameter);
        auto order = get_transpose_order(value);
        if (order.size() > 0 && !std::is_sorted(order.begin(), order.end())) {
            OV_FRONTEND_REQUIRE(ps.is_static());
            auto sh = ps.get_shape();
            ngraph::Shape new_shape(sh.size());
            for (int i = 0; i < sh.size(); i++) {
                new_shape[order[i]] = sh[i];
            }
            auto shape_const = ngraph::opset7::Constant::create(ngraph::element::i64, { new_shape.size() }, new_shape);
            auto reshape = std::make_shared<ngraph::opset7::Reshape>(parameter, shape_const, false);
            auto order_const = ngraph::opset7::Constant::create(ngraph::element::i32, { order.size() }, order);
            auto transpose = std::make_shared<ngraph::opset7::Transpose>(reshape, order_const);
            tensor_map[value->unique()] = transpose;
        } else {
            tensor_map[value->unique()] = parameter;
        }
    }

    // FIXME: select nodes in top-level block for now
    for (auto* node : graph->block()->nodes()) {
        //std::cout << "Node convert start" << std::endl;

        auto converted_outputs = convert_node(graph, node, tensor_map, rt_inputs);
        //std::cout << "Node convert before outputs" << std::endl;

        auto fw_outputs = node->outputs();

        // TODO: Make sure that mapping of fw_outputs to converted_outputs does always work
        // FIXME: Now it is not true for at least prim::Constant
        for(size_t i = 0; i < converted_outputs.size(); ++i) {
            size_t fw_tensor_id = fw_outputs[i]->unique();
            if(tensor_map.find(fw_tensor_id) != tensor_map.end()) {
                //std::cerr << "Duplicated producer for tensor with id = " << fw_tensor_id << " discovered at output "
                //    << "port " << i << " of node " << node->kind().toQualString() << "\n";
                throw "Duplicated producer";
            }

            // Output shape of converted node should match the original output shape
            //std::cerr << "[ DEBUG ] PT output shape = " << get_ov_shape(fw_outputs[i]) << '\n';
            //std::cerr << "[ DEBUG ] OV output shape = " << converted_outputs[i].get_partial_shape() << '\n';
            //OV_FRONTEND_REQUIRE(get_ov_shape(fw_outputs[i]) == converted_outputs[i].get_partial_shape());

            tensor_map[fw_tensor_id] = converted_outputs[i];
        }
        //std::cout << "Node convert end" << std::endl;
    }
    //std::cout << "All nodes convert end" << std::endl;

    ngraph::ResultVector results;
    //std::cerr << "Outputs:\n";
    for (auto* value: graph->outputs()) {
        //std::cerr << value->unique() << "\n";
        size_t id = value->unique();
        auto ov_output = tensor_map[id];
        auto order = get_transpose_order(value);
        if (order.size() > 0 && !std::is_sorted(order.begin(), order.end())) {
            throw "Output strides have wrong order.";
        }
        auto result = std::make_shared<ngraph::opset7::Result>(ov_output);
        results.push_back(result);
        //std::cerr << "Cluster result " << value->unique() << " with shape " << result->get_output_partial_shape(0) << "\n";
    }

    for(size_t i = 0; i < parameters.size(); ++i) {
        auto parameter = parameters[i];
        //std::cerr << "parameter[" << i << "].shape = "
        //    << parameter->get_output_shape(0) << ", consumers: " << parameter->output(0).get_target_inputs().size() << "\n";
    }
    //std::cout << "Convert end" << std::endl;

    return std::make_shared<ngraph::Function>(results, parameters);
}

}
}
}
}