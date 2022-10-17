#include "openvino/op/util/op_types.hpp"
#include "openvino/opsets/opset9.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/common_util.hpp"
#include "openvino/util/log.hpp"
#include "itt.hpp"
#include <ngraph/opsets/opset9.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pattern/op/or.hpp>
#include <transformations/utils/utils.hpp>

#include "transformations/common_optimizations/transpose_sinking_binary.hpp"

#include <utility>

namespace {

using NodePtr = std::shared_ptr<ov::Node>;
using Nodes = std::vector<NodePtr>;

ov::OutputVector GetOutputs(const Nodes & nodes) {
    ov::OutputVector outputs;

    for (auto & node : nodes)
        for (auto & output : node->outputs())
            outputs.push_back(output);

    return outputs;
}

// --------------------------------------------------------------------------------------

template <typename Predicate>
int FindFirstInputIf(const ov::Node * node, Predicate predicate) {
    for (int input_idx = 0; input_idx < node->get_input_size(); ++input_idx) {
        const ov::Node * input_node = node->get_input_node_ptr(input_idx);
        if (predicate(input_node))
            return input_idx;
    }

    return -1;
}

class IfTranpose {
public:
    IfTranpose() = default;
    bool operator()(const ov::Node * node) const {
        return dynamic_cast<const ov::opset9::Transpose*>(node) != nullptr;
    }
};

bool IfConcatHasTransposeInputs(const ov::Output<ov::Node>& output) {
    return FindFirstInputIf(output.get_node(), IfTranpose()) >= 0;
}

NodePtr GetFirstTransposeInput(const ov::Node * node) {
    const int index = FindFirstInputIf(node, IfTranpose());
    if (index < 0)
        return {};

    return node->get_input_node_shared_ptr(index);
}

// --------------------------------------------------------------------------------------

struct GraphBuildStrategyEmpty {
    GraphBuildStrategyEmpty() = default;
    GraphBuildStrategyEmpty(GraphBuildStrategyEmpty &&) = default;

    GraphBuildStrategyEmpty(const GraphBuildStrategyEmpty&) = delete;
    GraphBuildStrategyEmpty& operator=(const GraphBuildStrategyEmpty &) = delete;
};

template <typename ParentT>
class GraphBuildStrategy {
public:
    GraphBuildStrategy(ParentT prev_builder) :
        _prev_builder(std::forward<ParentT>(prev_builder)),
        _are_new_nodes_collected(false) {}

    GraphBuildStrategy(GraphBuildStrategy &&) = default;
    virtual ~GraphBuildStrategy() = default;
    GraphBuildStrategy & operator=(GraphBuildStrategy &&) = default;

    GraphBuildStrategy(const GraphBuildStrategy &) = delete;
    GraphBuildStrategy & operator=(const GraphBuildStrategy &) = delete;

    void build(Nodes & level_nodes)
    {
        _prev_builder.build(level_nodes);
        buildLevel(level_nodes);
    }

    Nodes getNewNodes() const {
        Nodes nodes = _prev_builder.getNewNodes();
        nodes.insert(nodes.end(), _new_nodes.begin(), _new_nodes.end());
        return nodes;
    }

    void addNewNode(NodePtr node) {
        if (_are_new_nodes_collected)
            _new_nodes.push_back(node);
    }

    void SetNewNodesCollected() { _are_new_nodes_collected = true; }

protected:
    virtual void buildLevel(Nodes & level_nodes) = 0;

private:
    ParentT _prev_builder;
    Nodes _new_nodes;
    bool _are_new_nodes_collected;
};

template <>
void GraphBuildStrategy<GraphBuildStrategyEmpty>::build(Nodes& level_nodes) {
    buildLevel(level_nodes);
}

template <>
Nodes GraphBuildStrategy<GraphBuildStrategyEmpty>::getNewNodes() const {
    return _new_nodes;
}

template <typename ParentT>
class CloneNodeStrategy : public GraphBuildStrategy<ParentT> {
public:
    CloneNodeStrategy(NodePtr node, ParentT prev_builder) :
        GraphBuildStrategy<ParentT>(std::forward<ParentT>(prev_builder)),
        _node(node) {}

    CloneNodeStrategy(CloneNodeStrategy &&) = default;
    CloneNodeStrategy& operator=(CloneNodeStrategy &&) = default;
    ~CloneNodeStrategy() = default;

    CloneNodeStrategy(const CloneNodeStrategy&) = delete;
    CloneNodeStrategy& operator=(const CloneNodeStrategy&) = delete;

protected:
    void buildLevel(Nodes & level_nodes) override
    {
        NodePtr new_layer = _node->clone_with_new_inputs(GetOutputs(level_nodes));
        ov::copy_runtime_info(_node, new_layer);

        level_nodes.resize(1);
        level_nodes[0] = new_layer;

        GraphBuildStrategy<ParentT>::addNewNode(new_layer);
    }

private:
    NodePtr _node;
};

template <typename ParentT>
CloneNodeStrategy<ParentT> AppendClonedNode(NodePtr node, ParentT prev_builder)
{
    return CloneNodeStrategy<ParentT>(node, std::forward<ParentT>(prev_builder));
}

template <typename AppendPredicateF, typename NodeCreateF, typename ParentT>
class InsertIf : public GraphBuildStrategy<ParentT>
{
public:
    InsertIf(ParentT prev_builder,
             AppendPredicateF append_predicate_f,
             NodeCreateF node_create_f) :
                GraphBuildStrategy<ParentT>(std::forward<ParentT>(prev_builder)),
                _append_predicate_f(std::forward<AppendPredicateF>(append_predicate_f)),
                _node_create_f(std::forward<NodeCreateF>(node_create_f)) {}

    InsertIf(InsertIf &&) = default;
    ~InsertIf() = default;

    InsertIf(const InsertIf&) = delete;
    InsertIf& operator=(const InsertIf&) = delete;

    void buildLevel(Nodes & level_nodes) override
    {
        for (size_t idx = 0; idx < level_nodes.size(); ++idx) {
            if (!_append_predicate_f(idx))
                continue;
            for (auto new_node : _node_create_f(level_nodes, idx))
                GraphBuildStrategy<ParentT>::addNewNode(new_node);
        }
    }

private:
    AppendPredicateF _append_predicate_f;
    NodeCreateF _node_create_f;
};

class InsertTranspose {
public:
    InsertTranspose(const ov::AxisVector & transpose_axis_order,
                 ov::element::Type transpose_element_type) :
                 _transpose_axis_order(transpose_axis_order),
                 _transpose_element_type(transpose_element_type) {}

    InsertTranspose(InsertTranspose &&) = default;
    ~InsertTranspose() = default;

    InsertTranspose(const InsertTranspose&) = delete;
    InsertTranspose& operator=(const InsertTranspose&) = delete;

    Nodes operator()(Nodes & level_nodes, size_t parent_node_idx) const;
private:
    ov::AxisVector _transpose_axis_order;
    ov::element::Type _transpose_element_type;
};

Nodes InsertTranspose::operator()(Nodes & level_nodes, size_t parent_node_idx) const
{
    auto & parent_node = level_nodes[parent_node_idx];

    auto transpose_const = std::make_shared<ov::opset9::Constant>(_transpose_element_type,
                                                                  ov::Shape{_transpose_axis_order.size()},
                                                                  _transpose_axis_order);
    auto transpose = std::make_shared<ov::opset9::Transpose>(parent_node, transpose_const);

    ov::copy_runtime_info(parent_node, {transpose, transpose_const});

    level_nodes[parent_node_idx] = transpose;

    return Nodes{transpose, transpose_const};
}

template <typename AppendPredicateF, typename NodeCreateF, typename ParentT = GraphBuildStrategyEmpty>
InsertIf<AppendPredicateF, NodeCreateF, ParentT> AppendTransposes(NodeCreateF create_node_f,
                                                                  AppendPredicateF predicate,
                                                                  ParentT prev_builder = GraphBuildStrategyEmpty())
{
    return InsertIf<AppendPredicateF, NodeCreateF, ParentT>(std::forward<ParentT>(prev_builder),
                                                            std::forward<AppendPredicateF>(predicate),
                                                            std::forward<NodeCreateF>(create_node_f));
}

class IfNotIndex {
public:
    IfNotIndex(size_t idx) : _idx(idx) {}
    bool operator()(size_t idx) const {
        return idx != _idx;
    }

private:
    const size_t _idx;
};

class AnyIndex {
public:
    AnyIndex() = default;
    bool operator()(size_t) const {
        return true;
    }
};

// --------------------------------------------------------------------------------------

template <typename GraphBuildStrategyT>
Nodes DoTransformation(NodePtr last_node,
                      const Nodes & input_nodes,
                      GraphBuildStrategyT graph_builder)
{
    Nodes layer_nodes = input_nodes;

    graph_builder.build(layer_nodes);

    auto & new_last_node = layer_nodes[0];

    new_last_node->set_friendly_name(last_node->get_friendly_name());
    ov::replace_node(last_node, new_last_node);

    return graph_builder.getNewNodes();
}

size_t GetNodeInputIndex(NodePtr node, NodePtr input_node)
{
    for (auto & output : input_node->outputs()) {
        for (auto & input : output.get_target_inputs()) {
            if (input.get_node()->get_instance_id() == node->get_instance_id())
                return input.get_index();
        }   
    }

    throw std::runtime_error("input node index not found");

    return 0;
}

ov::AxisVector GetTransposeOrder(NodePtr transpose)
{
    auto transpose_constant_node = ov::as_type_ptr<ov::opset9::Constant>(transpose->input_value(1).get_node_shared_ptr());
    return transpose_constant_node->get_axis_vector_val();
}

ov::AxisVector ReverseTransposeOrder(const ov::AxisVector & axis_order)
{
    ov::AxisVector out(axis_order.size());
    for (size_t i = 0; i < axis_order.size(); i++) {
        out.at(axis_order[i]) = i;
    }
    return out;
}

Nodes GetNodes(const std::vector<ov::Output<ov::Node>> & outputs)
{
    Nodes nodes;
    std::transform(outputs.begin(), outputs.end(), std::back_inserter(nodes),
                    [](ov::Output<ov::Node> output) { return output.get_node_shared_ptr(); });
    return nodes;
}

ov::element::Type GetTransposeElementType(NodePtr node)
{
    auto const_node = node->get_input_node_shared_ptr(1);
    if (!const_node)
        return {};

    auto const_node_const = std::dynamic_pointer_cast<ov::opset9::Constant>(const_node);
    if (!const_node_const)
        return {};

    return const_node_const->get_element_type();
}

}

// --------------------------------------------------------------------------------------

ngraph::pass::TransposeSinkingBinaryForward::TransposeSinkingBinaryForward() {
    MATCHER_SCOPE(TransposeSinkingBinaryForward);

    auto transpose_label =
        ov::pass::pattern::wrap_type<ov::opset9::Transpose>({ov::pass::pattern::any_input(),
                                                             ov::pass::pattern::wrap_type<ov::opset9::Constant>()},
                                                             ov::pass::pattern::consumers_count(1)); // FIXME: constraints (unit tests on constraints ?)
    auto binary_label_left = ov::pass::pattern::wrap_type<ov::op::util::BinaryElementwiseArithmetic>({transpose_label,
                                                                                                      ov::pass::pattern::any_input()}); // FIXME: constraints (unit tests on constraints ?)

    auto binary_label_right = ov::pass::pattern::wrap_type<ov::op::util::BinaryElementwiseArithmetic>({ov::pass::pattern::any_input(),
                                                                                                 transpose_label}); // FIXME: constraints (unit tests on constraints ?)
    auto binary_label = std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{binary_label_left, binary_label_right});

    ov::matcher_pass_callback matcher_pass_callback = [=](ov::pass::pattern::Matcher& m) {
        auto binary = m.get_match_root();
        auto transpose = GetFirstTransposeInput(binary.get());

        const ov::AxisVector transpose_axis_order = GetTransposeOrder(transpose);
        const ov::AxisVector reversed_traspose_axis_order = ReverseTransposeOrder(transpose_axis_order);
        const size_t tranpose_input_index = GetNodeInputIndex(binary, transpose);
        const ov::element::Type transpose_element_type = GetTransposeElementType(transpose);

        // Graph build strategy
        auto insert_reversed_transpose = InsertTranspose(reversed_traspose_axis_order, transpose_element_type);
        auto append_input_transposes = AppendTransposes(std::move(insert_reversed_transpose), IfNotIndex(tranpose_input_index));

        auto clone_binary = AppendClonedNode(binary, std::move(append_input_transposes));

        auto insert_transpose = InsertTranspose(transpose_axis_order, transpose_element_type);
        auto append_output_transpose = AppendTransposes(std::move(insert_transpose), AnyIndex(), std::move(clone_binary));
        append_output_transpose.SetNewNodesCollected();
        //
        Nodes input_nodes = GetNodes(binary->input_values());
        input_nodes[tranpose_input_index] = transpose->input_value(0).get_node_shared_ptr();

        for (auto & new_node: DoTransformation(binary, input_nodes, std::move(append_output_transpose))) {
            register_new_node(new_node);
        }

        return true;
    };

    auto matcher = std::make_shared<ov::pass::pattern::Matcher>(binary_label,matcher_name);
    register_matcher(matcher, matcher_pass_callback);
}

ngraph::pass::TransposeSinkingBinaryBackward::TransposeSinkingBinaryBackward() {
    MATCHER_SCOPE(TransposeSinkingBinaryBackward);

    auto binary_label = ov::pass::pattern::wrap_type<ov::op::util::BinaryElementwiseArithmetic>({ov::pass::pattern::any_input(),
                                                                                                 ov::pass::pattern::any_input()}); // FIXME: constraints (unit tests on constraints ?)

    auto transpose_label =
        ov::pass::pattern::wrap_type<ov::opset9::Transpose>({binary_label,
                                                             ov::pass::pattern::wrap_type<ov::opset9::Constant>()},
                                                             ov::pass::pattern::consumers_count(1)); // FIXME: constraints (unit tests on constraints ?)

    ov::matcher_pass_callback matcher_pass_callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto transpose = pattern_to_output.at(transpose_label).get_node_shared_ptr();
        auto binary = pattern_to_output.at(binary_label).get_node_shared_ptr();

        const ov::AxisVector transpose_axis_order = GetTransposeOrder(transpose);
        const ov::element::Type transpose_element_type = GetTransposeElementType(transpose);

        // Graph build strategy
        auto insert_input_transpose = InsertTranspose(transpose_axis_order, transpose_element_type);
        auto append_input_transposes = AppendTransposes(std::move(insert_input_transpose), AnyIndex());
        append_input_transposes.SetNewNodesCollected();

        auto clone_binary = AppendClonedNode(binary, std::move(append_input_transposes));
        //

        const Nodes input_nodes = GetNodes(binary->input_values());

        for (auto & new_node: DoTransformation(transpose, input_nodes, std::move(clone_binary))) {
            register_new_node(new_node);
        }

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(transpose_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}

// --------------------------------------------------------------------------------------

namespace {

template <typename ParentT>
class AppendConcatStrategy : public GraphBuildStrategy<ParentT> {
public:
    AppendConcatStrategy(int64_t axis, ParentT prev_builder) :
        GraphBuildStrategy<ParentT>(std::forward<ParentT>(prev_builder)),
        _axis(axis) {}

    ~AppendConcatStrategy() = default;
    AppendConcatStrategy(AppendConcatStrategy &&) = default;
    AppendConcatStrategy& operator=(AppendConcatStrategy &&) = default;

    AppendConcatStrategy& operator=(const AppendConcatStrategy&) = delete;
    AppendConcatStrategy(const AppendConcatStrategy&) = delete;

    void buildLevel(Nodes & level_nodes) override
    {
        NodePtr new_layer = std::make_shared<ov::opset9::Concat>(level_nodes, _axis);
        ov::copy_runtime_info(level_nodes[0], new_layer);

        level_nodes.resize(1);
        level_nodes[0] = new_layer;

        GraphBuildStrategy<ParentT>::addNewNode(new_layer);
    }

private:
    const int64_t _axis;
};

template <typename ParentT>
AppendConcatStrategy<ParentT> AppendConcat(int64_t axis, ParentT prev_builder)
{
    return AppendConcatStrategy<ParentT>(axis, std::forward<ParentT>(prev_builder));
}

int64_t GetConcatAxis(NodePtr concat_node)
{
    auto concat = ov::as_type_ptr<ov::opset9::Concat>(concat_node);
    if (!concat)
        return -1;

    return concat->get_axis();
}

// get new axis for concat according to transpose order
int64_t TransposeConcatAxis(int64_t axis, const ov::AxisVector & transpose_order)
{
    return transpose_order[axis];
}

} // namespace

ngraph::pass::TransposeSinkingConcatForward::TransposeSinkingConcatForward() {
    MATCHER_SCOPE(TransposeSinkingConcatForward);

    auto concat_label = ov::pass::pattern::wrap_type<ov::opset9::Concat>(IfConcatHasTransposeInputs);

    ov::matcher_pass_callback matcher_pass_callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto concat_output = pattern_to_output.at(concat_label);
        auto concat = concat_output.get_node_shared_ptr();
        auto transpose = GetFirstTransposeInput(concat.get());

        const ov::AxisVector transpose_axis_order = GetTransposeOrder(transpose);
        const ov::AxisVector reversed_traspose_axis_order = ReverseTransposeOrder(transpose_axis_order);
        const size_t tranpose_input_index = GetNodeInputIndex(concat, transpose);
        const ov::element::Type transpose_element_type = GetTransposeElementType(transpose);
        const int64_t transposed_concat_axis = TransposeConcatAxis(GetConcatAxis(concat), transpose_axis_order);

        // Graph build strategy
        auto insert_revered_transpose = InsertTranspose(reversed_traspose_axis_order, transpose_element_type);
        auto append_input_transposes = AppendTransposes(std::move(insert_revered_transpose), IfNotIndex(tranpose_input_index));

        auto append_concat = AppendConcat(transposed_concat_axis, std::move(append_input_transposes));

        auto add_output_transpose = InsertTranspose(transpose_axis_order, transpose_element_type);
        auto append_output_transpose = AppendTransposes(std::move(add_output_transpose), AnyIndex(), std::move(append_concat));
        append_output_transpose.SetNewNodesCollected();
        //

        Nodes input_nodes = GetNodes(concat->input_values());
        input_nodes[tranpose_input_index] = transpose->input_value(0).get_node_shared_ptr();

        for (auto & new_node: DoTransformation(concat, input_nodes, std::move(append_output_transpose))) {
            register_new_node(new_node);
        }

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(concat_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}

ngraph::pass::TransposeSinkingConcatBackward::TransposeSinkingConcatBackward() {
    MATCHER_SCOPE(TransposeSinkingConcatBackward);

    auto concat_label = ov::pass::pattern::wrap_type<ov::opset9::Concat>();

    auto transpose_label =
        ov::pass::pattern::wrap_type<ov::opset9::Transpose>({concat_label,
                                                             ov::pass::pattern::wrap_type<ov::opset9::Constant>()},
                                                             ov::pass::pattern::consumers_count(1)); // FIXME: constraints (unit tests on constraints ?)

    ov::matcher_pass_callback matcher_pass_callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto transpose = pattern_to_output.at(transpose_label).get_node_shared_ptr();
        auto concat = pattern_to_output.at(concat_label).get_node_shared_ptr();

        const ov::AxisVector transpose_axis_order = GetTransposeOrder(transpose);
        const ov::AxisVector reversed_traspose_axis_order = ReverseTransposeOrder(transpose_axis_order);
        const ov::element::Type transpose_element_type = GetTransposeElementType(transpose);
        const int64_t transposed_concat_axis = TransposeConcatAxis(GetConcatAxis(concat), reversed_traspose_axis_order);

        // Graph build strategy
        auto insert_transpose = InsertTranspose(transpose_axis_order, transpose_element_type);
        auto append_input_transposes = AppendTransposes(std::move(insert_transpose), AnyIndex());
        append_input_transposes.SetNewNodesCollected();

        auto append_concat = AppendConcat(transposed_concat_axis, std::move(append_input_transposes));
        //

        const Nodes input_nodes = GetNodes(concat->input_values());

        for (auto & new_node: DoTransformation(transpose, input_nodes, std::move(append_concat))) {
            register_new_node(new_node);
        }

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(transpose_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}
