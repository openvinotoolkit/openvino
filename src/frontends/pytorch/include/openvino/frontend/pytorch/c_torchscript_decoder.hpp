// This is torch jit dependent piece of code for decoding TS jit graph inside Torch runtime
// This code was copied from inside PT source tree from POC branch, it cannot be compiled withou torch dependencies

#pragma once

#include <exception>
#include <map>
#include <memory>

// use new API
#include <ngraph/opsets/opset7.hpp>
#include <ngraph/opsets/opset8.hpp>

#include "decoder.hpp"
#include "openvino/op/util/framework_node.hpp"

#define OV_FRONTEND_REQUIRE(X)                                                \
    do                                                                        \
        if (!(X)) {                                                           \
            throw std::runtime_error(std::string("[ ERROR ] Failed: ") + #X); \
        }                                                                     \
    while (false)

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
        return ngraph::element::i8;  // TODO: Is it correct?
    case at::ScalarType::QUInt8:
        return ngraph::element::u8;  // TODO: Is it correct?
    default:
        TORCH_CHECK(false, "Not support data type ", dt);
    }
}

ngraph::PartialShape get_ov_shape(const Value* v) {
    std::vector<ngraph::Dimension> dims;
    if (v->type()->isSubtypeOf(TensorType::get())) {
        const auto& tt = v->type()->cast<TensorType>();
        // std::cerr << "sizes = " << tt->sizes() << "\n";
        // std::cerr << "dims = " << tt->dim() << "\n";

        const auto& sizes = tt->sizes();
        if (sizes.sizes())
            for (auto& d : *sizes.sizes())
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
            // std::cerr << "[ ERROR ] Cannot retrieve shape for not a tensor value\n";
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
        // std::cerr << "[ ERROR ] Cannot retrieve type for not a tensor value\n";
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
        std::stable_sort(idx.begin(), idx.end(), [&s](size_t lhs, size_t rhs) {
            return s[lhs] > s[rhs];
        });
    } else {
        // std::cerr << "[ ERROR ] Cannot retrieve strides for not a tensor value\n";
        return {};
    }
    return idx;
}

void mark_node(std::shared_ptr<ngraph::Node> ov_node, const Node* pt_node) {
    auto& rt_info = ov_node->get_rt_info();
    auto rt_info_pt = rt_info.find("pt_node");
    if (rt_info_pt == rt_info.end()) {
        rt_info_pt =
            rt_info.insert(std::map<std::string, ov::Any>::value_type("pt_node", std::set<const Node*>())).first;
    }
    rt_info_pt->second.as<std::set<const Node*>>().insert(pt_node);
}

typedef std::map<size_t, ngraph::Output<ngraph::Node>> TensorMap;

// TODO: This class should be reworked for new Decoder API
class NodeDecoder : public std::enable_shared_from_this<NodeDecoder> {
public:
    NodeDecoder(std::shared_ptr<Graph> _graph,
                const Node* _node,
                const TensorMap& _tensor_map,
                const TensorArgs& _graph_tensors)
        : graph(_graph),
          node(_node),
          tensor_map(_tensor_map),
          graph_tensors(_graph_tensors) {}

    // Do not search for input in tensor map; try to access it as a constant of specified type T and return its value
    template <typename T>
    T const_input(size_t index) const;

    // Search for input in tensor map and return an output port for already converted op
    ngraph::Output<ngraph::Node> input(size_t index) const {
        // std::cerr << "Trying to map input to ngraph...";
        OV_FRONTEND_REQUIRE(!input_is_none(index));
        return tensor_map.at(node->inputs()[index]->unique());
    }

    ngraph::OutputVector inputs() const {
        ngraph::OutputVector res;
        for (auto& input : node->inputs()) {
            // std::cerr << "Searching for input: " << input->unique() << "\n";
            OV_FRONTEND_REQUIRE(tensor_map.find(input->unique()) != tensor_map.end());
            res.push_back(tensor_map.at(input->unique()));
        }
        return res;
    }

    bool input_is_none(size_t index) const {
        return node->inputs().size() <= index || !node->inputs()[index]->mustNotBeNone();
    }

    // Convert the resulting value of this node to ngraph Constant; works correctly only for nodes that produce
    // constant value, naturally for prim::Constant
    ngraph::OutputVector as_constant();

    torch::jit::NodeKind get_op_type() const {
        return node->kind();
    }

    size_t num_of_outputs() const {
        return node->outputs().size();
    }

    c10::ArrayRef<const torch::jit::Value*> outputs() const {
        return node->outputs();
    }

    std::shared_ptr<ngraph::Node> mark_node(std::shared_ptr<ngraph::Node> ov_node) const {
        openvino::mark_node(ov_node, node);
        return ov_node;
    }

    void mark_nodes(std::vector<std::shared_ptr<ngraph::Node>> ov_nodes) const {
        for (auto& ov_node : ov_nodes) {
            openvino::mark_node(ov_node, node);
        }
    }

    ngraph::Output<ngraph::Node> mark_output(ngraph::Output<ngraph::Node> ov_output) const {
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
std::vector<Tdst> convert_vector(const std::vector<Tsrc>& in) {
    return std::vector<Tdst>(in.begin(), in.end());
}

template <>
std::vector<int64_t> NodeDecoder::const_input<std::vector<int64_t>>(size_t index) const {
    OV_FRONTEND_REQUIRE(!input_is_none(index));
    return toIValue(node->input(index))->toIntVector();
}

template <>
std::string NodeDecoder::const_input<std::string>(size_t index) const {
    OV_FRONTEND_REQUIRE(!input_is_none(index));
    return toIValue(node->input(index))->toString()->string();
}

template <>
ngraph::Strides NodeDecoder::const_input<ngraph::Strides>(size_t index) const {
    return convert_vector<ngraph::Strides::value_type>(const_input<std::vector<int64_t>>(index));
}

template <>
ngraph::CoordinateDiff NodeDecoder::const_input<ngraph::CoordinateDiff>(size_t index) const {
    return convert_vector<ngraph::CoordinateDiff::value_type>(const_input<std::vector<int64_t>>(index));
}

template <>
ngraph::Shape NodeDecoder::const_input<ngraph::Shape>(size_t index) const {
    return convert_vector<ngraph::Shape::value_type>(const_input<std::vector<int64_t>>(index));
}

template <>
int64_t NodeDecoder::const_input<int64_t>(size_t index) const {
    OV_FRONTEND_REQUIRE(!input_is_none(index));
    return toIValue(node->input(index))->toInt();
}

template <>
bool NodeDecoder::const_input<bool>(size_t index) const {
    OV_FRONTEND_REQUIRE(!input_is_none(index));
    return toIValue(node->input(index))->toBool();
}

template <>
double NodeDecoder::const_input<double>(size_t index) const {
    OV_FRONTEND_REQUIRE(!input_is_none(index));
    return toIValue(node->input(index))->toDouble();
}

template <>
float NodeDecoder::const_input<float>(size_t index) const {
    return const_input<double>(index);
}

ngraph::Output<ngraph::Node> NodeDecoder::make_constant(const at::Tensor& tensor) const {
    auto shape = convert_vector<ngraph::Shape::value_type>(tensor.sizes().vec());
    // TODO: Check strides; now, for our first experiments we expect that they are trivial and don't introduce gaps
    // auto strides = t.strides().vec();
    auto type = to_ngraph_data_type(tensor.scalar_type());
    return std::make_shared<ngraph::opset7::Constant>(type, shape, tensor.data_ptr());
}

ngraph::OutputVector NodeDecoder::as_constant() {
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
        std::cerr << "[ ERROR ] Constant type " << value->tagKind()
                  << " is not recognized; replaced by PT FrameworkNode\n";
        // Instead of doing PtFrameworkNode make a constant of custom type -- more code is expected here
        // constant = std::make_shared<PtFrameworkNode>(shared_from_this(), ngraph::OutputVector{});
    }
    return {mark_node(constant)};
}
}
}
}
