// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#if 0
#include <transformations/low_precision/main.hpp>

#include <memory>

#include <ngraph/pass/manager.hpp>
#include <transformations/low_precision/concat_multi_channels.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset2.hpp>
#include <details/ie_exception.hpp>

namespace ngraph {
namespace pass {
namespace low_precision {

// bool is_child(std::vector<Output<Node>> node_outputs, const std::vector<NodeTypeInfo>& layer_types, const std::vector<NodeTypeInfo>& ignore_types);


// bool is_child(std::vector<Output<Node>> node_outputs, const std::vector<NodeTypeInfo>& layer_types, const std::vector<NodeTypeInfo> ignore_types) {
//     std::vector<std::shared_ptr<Node>> nodes(node_outputs.size());
//     std::transform(node_outputs.begin(), node_outputs.end(), nodes.begin(), [](Output<Node> port) { return port.get_node_shared_ptr(); });
//     return is_child(nodes, layer_types, ignore_types);
// }

template <typename Operation, typename Callback>
void add_single_node_pattern(ngraph::pass::GraphRewrite* transformation, Callback callback) {
    using namespace ngraph;

    auto is_op_type = [](std::shared_ptr<Node> n) {
        return !!as_type_ptr<Operation>(n);
    };

    auto p_node = std::make_shared<pattern::op::Label>(element::f32, Shape{}, is_op_type);

    ngraph::graph_rewrite_callback internal_callback = [callback](ngraph::pattern::Matcher &m) {
        auto l_node = dynamic_pointer_cast<Operation>(m.get_match_root());
        if (!l_node) {
            std::cerr << "Error my matcher 1!!!\n";
            return false;
        }

        std::cerr << "TEST2: Type = " << l_node->get_type_info().name << ", name = " << l_node->get_friendly_name() << "\n";

        return callback(l_node);
    };
    auto m = std::make_shared<ngraph::pattern::Matcher>(p_node, "SingleNodeMatcher");
    transformation->add_matcher(m, internal_callback, ngraph::pass::PassProperty::CHANGE_DYNAMIC_STATE);
}


bool getQuantizeLayers(
        std::shared_ptr<Node> layer,
        std::vector<std::string>& childNameOurAfterQuantizeLayers,
        std::vector<std::shared_ptr<opset1::FakeQuantize>>& quantizeLayers,
        std::vector<std::vector<std::shared_ptr<Node>>>& intermediateLayers,
        std::vector<std::shared_ptr<opset1::Concat>>& concatLayers,
        std::string childName,
        std::vector<std::shared_ptr<Node>>& sideOutputLayers,
        std::vector<std::string>& childrenNameSideOutputLayers) {
    const std::unordered_set<NodeTypeInfo> POOLINGS = {opset1::AvgPool::get_type_info_static(), opset1::MaxPool::get_type_info_static()};
    if (!layer->get_type_info().is_castable(opset1::FakeQuantize::get_type_info_static())) {
        do {
            if (is_castable_to_one_of(layer->get_type_info(), POOLINGS)) {
                intermediateLayers.back().push_back(layer);
                std::vector<std::shared_ptr<Node>> children = getChildrenRecursivelyExceptTypes(layer, POOLINGS);
                std::string concatName;
                for (auto child : children) {
                    if (child->get_type_info().is_castable(opset1::Concat::get_type_info_static())) {
                        if (!concatName.empty()) {
                            THROW_TRANSFORMATION_EXCEPTION << "several concat children layers are not supported";
                        }
                        concatName = child->get_friendly_name();
                    }
                }

                childName = concatName;
                layer = layer->get_input_node_shared_ptr(0);
            } else if (layer->get_type_info().is_castable(opset1::Concat::get_type_info_static())) {
                concatLayers.push_back(as_type_ptr<opset1::Concat>(layer));

                if (consumers(layer).size() != 1) {
                    sideOutputLayers.push_back(layer);
                    childrenNameSideOutputLayers.push_back(childName);
                }
                int size = layer->inputs().size();
                childName = layer->get_friendly_name();
                for (int i = 0; i < size; i++) {
                    std::shared_ptr<Node> layer1 = layer->get_input_node_shared_ptr(i);
                    intermediateLayers.push_back({});
                    if (!getQuantizeLayers(
                            layer1,
                            childNameOurAfterQuantizeLayers,
                            quantizeLayers,
                            intermediateLayers,
                            concatLayers,
                            childName,
                            sideOutputLayers,
                            childrenNameSideOutputLayers)) {
                        return false;
                    }
                }
                return true;
            } else {
                return false;
            }
        } while (!layer->get_type_info().is_castable(opset1::FakeQuantize::get_type_info_static()));
    }

    childNameOurAfterQuantizeLayers.push_back(childName);
    quantizeLayers.push_back(as_type_ptr<opset1::FakeQuantize>(layer));
    return true;
}



class TRANSFORMATIONS_API Branch : public Transformation {
public:
    Branch(TransformationContext& _context, const Transformation::Params& params) : Transformation(_context, params) {
        add_single_node_pattern<opset1::Concat>(this, [](std::shared_ptr<opset1::Concat> concat){
            if (concat->get_axis() != 1) {
                return false;
            }

            if (is_child(consumers(concat),
                         {opset1::Concat::get_type_info_static()},
                         {opset1::AvgPool::get_type_info_static(), opset1::MaxPool::get_type_info_static()})) {
                return false;
            }

            std::cerr << "PASSED NEW POINT 1: " << concat->get_friendly_name() << "\n";

            std::vector<std::shared_ptr<opset1::FakeQuantize>> quantizeLayers;
            std::vector<std::vector<std::shared_ptr<Node>>> intermediateLayers;
            std::vector<std::shared_ptr<opset1::Concat>> concatLayers;
            std::vector<std::string> childNameOurAfterQuantizeLayers;
            std::vector<std::shared_ptr<Node>> sideOutputLayers;
            std::vector<std::string> childrenNameSideOutputLayers;
            const auto inputDataNumber = concat->inputs().size();
            for (size_t index = 0lu; index < inputDataNumber; index++) {
                auto parentLayer = concat->get_input_node_shared_ptr(index);
                intermediateLayers.push_back({});
                if (!getQuantizeLayers(
                        parentLayer,
                        childNameOurAfterQuantizeLayers,
                        quantizeLayers,
                        intermediateLayers,
                        concatLayers,
                        concat->get_friendly_name(),
                        sideOutputLayers,
                        childrenNameSideOutputLayers)) {
                    return false;
                }
            }

            concatLayers.insert(concatLayers.begin(), concat);

            if (quantizeLayers.empty()) {
                return false;
            }

            std::cerr << "PASSED NEW POINT 2: " << quantizeLayers.size()
                      << ' ' << intermediateLayers.size() << ' ' << concatLayers.size() << ' '
                      << childNameOurAfterQuantizeLayers.size() << ' ' << sideOutputLayers.size() << ' '
                      << childrenNameSideOutputLayers.size() << '\n';

            for (auto quantizeLayer : quantizeLayers) {
                if (!QuantizationDetails::outputLayoutIsSupported(quantizeLayer)) {
                    return false;
                }
            }

            std::cerr << "PASSED NEW POINT 3: " << concat->get_friendly_name() << "\n";

            // TODO: Continue writing transformation code here. Up to this point we match flow to the old transformation.

            return true;
        });
    }
};



class TRANSFORMATIONS_API Layers : public Transformation {
public:
    Layers(TransformationContext& _context, const Transformation::Params& params) : Transformation(_context, params) {
        add_single_node_pattern<opset1::FakeQuantize>(this, [](std::shared_ptr<opset1::FakeQuantize>){ return true; });
        add_single_node_pattern<opset1::Convolution>(this, [](std::shared_ptr<opset1::Convolution>){
            // TODO: Move this override statement to externally accessible API
            // This overrides default parameter by one provided in MKLDNN plugin configuration
            std::vector<element::Type> precisionsOnActivations = {element::u8};

            return true;
        });
        add_single_node_pattern<opset1::AvgPool>(this, [](std::shared_ptr<opset1::AvgPool>){ return true; });
        add_single_node_pattern<opset1::MaxPool>(this, [](std::shared_ptr<opset1::MaxPool>){ return true; });
        add_single_node_pattern<opset1::FakeQuantize>(this, [](std::shared_ptr<opset1::FakeQuantize>){ return true; });
        add_single_node_pattern<opset1::Reshape>(this, [](std::shared_ptr<opset1::Reshape>){ return true; });
        add_single_node_pattern<opset1::MatMul>(this, [](std::shared_ptr<opset1::MatMul>){ return true; });
        add_single_node_pattern<opset1::Transpose>(this, [](std::shared_ptr<opset1::Transpose>){ return true; });
        add_single_node_pattern<opset1::Squeeze>(this, [](std::shared_ptr<opset1::Squeeze>){ return true; });
        add_single_node_pattern<opset1::Relu>(this, [](std::shared_ptr<opset1::Relu>){ return true; });
        add_single_node_pattern<opset2::MVN>(this, [](std::shared_ptr<opset2::MVN>){ return true; });
        add_single_node_pattern<opset1::Add>(this, [](std::shared_ptr<opset1::Add>){ return true; });
        add_single_node_pattern<opset1::Multiply>(this, [](std::shared_ptr<opset1::Multiply>){ return true; });
        add_single_node_pattern<opset1::Interpolate>(this, [](std::shared_ptr<opset1::Interpolate>){ return true; });
    }
};

class TRANSFORMATIONS_API Cleanup : public Transformation {
public:
    Cleanup(TransformationContext& _context, const Transformation::Params& params) : Transformation(_context, params) {
        add_single_node_pattern<opset1::FakeQuantize>(this, [](std::shared_ptr<opset1::FakeQuantize>){ return true; });
        add_single_node_pattern<opset1::Add>(this, [](std::shared_ptr<opset1::Add>){
            // TODO: Configure parameters similarly to Convolution
            return true;
        });
        add_single_node_pattern<opset1::Multiply>(this, [](std::shared_ptr<opset1::Multiply>){
            // TODO: Configure parameters similarly to Convolution
            return true;
        });
    }
};




bool ngraph::pass::LowPrecisionTransformations::run_on_function(std::shared_ptr<ngraph::Function> f) {
    using namespace ngraph::pass;
    Manager passes;

    auto params = low_precision::Transformation::Params(
            true,  // updatePrecisions
            true,  // quantizeOutputs
            true,  // weightsToConst
            low_precision::Transformation::QuantizedTensorAlignment::UpdateLevel,  // quantizedTensorAlignmentOnActivations
            low_precision::Transformation::QuantizedTensorAlignment::None,  // quantizedTensorAlignmentOnWeights
            true,  // roundQuantizedValues
            true,  // updateBiases
            true);  // supportAsymmetricQuantization

    // TODO: Add API for fine-tuning of each transformation stage below (currently embed this configuration
    // TODO: from MKLDNN plugin directly to matchers; this makes matchers tuned for CPU only, not GPU)

    // TODO: Check all FQ.level values are supported (simple)

    low_precision::TransformationContext context(f);
    // TODO: Really initialize context (simple)

    passes.register_pass<low_precision::Branch>(context, params);
    // TODO: New transformation required: Multiply/Add + FQ fusion; required by FQ transform below
    passes.register_pass<low_precision::FakeQuantize>(context, params);
    passes.register_pass<low_precision::Layers>(context, params);
    passes.register_pass<low_precision::Cleanup>(context, params);
    passes.run_passes(f);
    return true;
}

#endif