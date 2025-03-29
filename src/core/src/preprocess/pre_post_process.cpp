// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/preprocess/pre_post_process.hpp"

#include "color_utils.hpp"
#include "function_guard.hpp"
#include "itt.hpp"
#include "layout_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/op/true.hpp"
#include "preprocess_impls.hpp"
#include "transformations/common_optimizations/convolution_to_group_convolution_fusion.hpp"
#include "transformations/common_optimizations/disable_random_uniform_constant_folding.hpp"
#include "transformations/common_optimizations/disable_shapeof_constant_folding.hpp"
#include "transformations/common_optimizations/gelu_fusion.hpp"
#include "transformations/common_optimizations/mul_conv_fusion.hpp"
#include "transformations/common_optimizations/ric_fusion.hpp"
#include "transformations/common_optimizations/shared_ops_optimization.hpp"
#include "transformations/fp16_compression/mark_decompression_convert_constant_folding.hpp"
#include "transformations/low_precision/mark_dequantization_subgraph.hpp"
#include "transformations/op_conversions/convert_divide.hpp"
#include "transformations/rt_info/dequantization_node.hpp"
#include "transformations/utils/utils.hpp"

namespace {

struct RTInfoCache {
    template <typename Func>
    void traverse(const std::shared_ptr<ov::Model>& model, Func&& func) {
        for (const auto& op : model->get_ordered_ops()) {
            func(op);
            if (const auto& multi_subgraph_op = ov::as_type_ptr<ov::op::util::MultiSubGraphOp>(op)) {
                for (const auto& sub_graph : multi_subgraph_op->get_functions()) {
                    if (sub_graph)
                        traverse(sub_graph, func);
                }
            }
        }
    }

    void store(const std::shared_ptr<ov::Model>& model) {
        traverse(model, [this](const std::shared_ptr<ov::Node>& op) {
            m_rt_info_cache[op.get()] = op->get_rt_info();
        });
    }

    void restore(const std::shared_ptr<ov::Model>& model) {
        traverse(model, [this](const std::shared_ptr<ov::Node>& op) {
            auto it = m_rt_info_cache.find(op.get());
            if (it != m_rt_info_cache.end()) {
                op->get_rt_info() = it->second;
            } else {
                ov::pass::enable_constant_folding(op);
                ov::unmark_dequantization_node(op);
                ov::unmark_as_decompression(op);
            }
        });
    }

    std::unordered_map<ov::Node*, ov::RTMap> m_rt_info_cache;
};

void transformation_pipeline(std::shared_ptr<ov::Model>& model) {
    using namespace ov;
    using namespace ov::pass;
    using namespace ov::element;

    // 0. Store RT info to not affect plugin compilation
    RTInfoCache rt_info_cache;
    rt_info_cache.store(model);

    Manager manager("pre_post_processing");
    manager.set_per_pass_validation(false);

    // prerequisite: the model structure optimization before applying of the markup
    REGISTER_PASS(manager, SharedOpOptimization)

    // 1. Set "disable_const_folding" attribute
    REGISTER_PASS(manager, MarkDequantization, TypeVector{i8, u8, i4, u4, nf4, f4e2m1, f8e4m3, f8e5m2, f8e8m0});
    REGISTER_PASS(manager, DisableShapeOfConstantFolding, false);
    REGISTER_PASS(manager, DisableRandomUniformConstantFolding)
    // Mark quantized and f16/bf16 compressed constants to prevent CF for them,
    // so that not extra memory is used for intermediate decompressed constants.
    REGISTER_PASS(manager, MarkCompressedFloatConstants);
    REGISTER_PASS(manager, DisableDecompressionConvertConstantFolding);

    // 2. Fusion transformations:
    REGISTER_PASS(manager, ConvertDivideWithConstant)
    auto fusions = manager.register_pass<GraphRewrite>();
    // Gelu fusion have to be executed before MulConv fusion because Mul(X, 0.5) might be fused to Conv weights
    ADD_MATCHER(fusions, GeluFusion)
    ADD_MATCHER(fusions, MultiplyConvolutionFusion)
    ADD_MATCHER(fusions, MultiplyGroupConvolutionFusion)
    ADD_MATCHER(fusions, MultiplyConvolutionBackpropDataFusion)
    ADD_MATCHER(fusions, MultiplyGroupConvolutionBackpropDataFusion)
    fusions->set_name("ov::pass::MultiplyFusions");
    REGISTER_PASS(manager, ReverseInputChannelsFusion)

    // 3. CF call due to detected perf degradations
    REGISTER_PASS(manager, ConstantFolding)
    manager.run_passes(model);

    // 4. Restore old RT info to not affect plugin compilation
    rt_info_cache.restore(model);
}

}  // namespace

namespace ov {
namespace preprocess {

//-------------- InputInfo ------------------
InputInfo::InputInfo() : m_impl(std::unique_ptr<InputInfoImpl>(new InputInfoImpl)) {}
InputInfo::InputInfo(InputInfo&& other) noexcept = default;
InputInfo& InputInfo::operator=(InputInfo&& other) noexcept = default;

InputInfo::~InputInfo() = default;

InputTensorInfo& InputInfo::tensor() {
    return m_impl->m_tensor_info;
}

PreProcessSteps& InputInfo::preprocess() {
    return m_impl->m_preprocess;
}

InputModelInfo& InputInfo::model() {
    return m_impl->m_model_data;
}

//-------------- OutputInfo ------------------
OutputInfo::OutputInfo() : m_impl(std::unique_ptr<OutputInfoImpl>(new OutputInfoImpl)) {}
OutputInfo::OutputInfo(OutputInfo&& other) noexcept = default;
OutputInfo& OutputInfo::operator=(OutputInfo&& other) noexcept = default;
OutputInfo::~OutputInfo() = default;

OutputModelInfo& OutputInfo::model() {
    return m_impl->m_model_info;
}

PostProcessSteps& OutputInfo::postprocess() {
    return m_impl->m_postprocess;
}

OutputTensorInfo& OutputInfo::tensor() {
    return m_impl->m_tensor_info;
}

// ------------------------ PrePostProcessor --------------------
struct PrePostProcessor::PrePostProcessorImpl {
public:
    PrePostProcessorImpl() = default;
    explicit PrePostProcessorImpl(const std::shared_ptr<ov::Model>& f) : m_function(f) {
        OPENVINO_ASSERT(f, "Model can't be nullptr for PrePostProcessor");

        // if IR version < 11, set compatibility mode
        const auto names_mode = m_function->has_rt_info("version") && m_function->get_rt_info<int64_t>("version") < 11;

        for (size_t i = 0; i < m_function->inputs().size(); ++i) {
            auto info = InputInfo();
            info.m_impl->m_resolved_param = m_function->get_parameters()[i];
            m_inputs.push_back(std::move(info));
        }
        for (size_t i = 0; i < m_function->outputs().size(); ++i) {
            auto info = OutputInfo();
            info.m_impl->m_output_node = m_function->output(i);
            info.m_impl->get_tensor_data()->set_names_compatibility_mode(names_mode);
            m_outputs.push_back(std::move(info));
        }
    }

    InputInfo& find_input(const std::string& tensor_name) {
        size_t index;
        for (index = 0; index < m_function->inputs().size(); index++) {
            if (m_function->input(index).get_tensor().get_names().count(tensor_name)) {
                return m_inputs[index];
            }
        }
        OPENVINO_ASSERT(false, "Model doesn't have input with name ", tensor_name);
    }

    OutputInfo& find_output(const std::string& tensor_name) {
        size_t index;
        for (index = 0; index < m_function->outputs().size(); index++) {
            if (m_function->output(index).get_tensor().get_names().count(tensor_name)) {
                return m_outputs[index];
            }
        }
        OPENVINO_ASSERT(false, "Model doesn't have output with name ", tensor_name);
    }

    std::vector<InputInfo> m_inputs;
    std::vector<OutputInfo> m_outputs;
    std::shared_ptr<Model> m_function = nullptr;
};

PrePostProcessor::PrePostProcessor(const std::shared_ptr<Model>& function)
    : m_impl(std::unique_ptr<PrePostProcessorImpl>(new PrePostProcessorImpl(function))) {}
PrePostProcessor::PrePostProcessor(PrePostProcessor&&) noexcept = default;
PrePostProcessor& PrePostProcessor::operator=(PrePostProcessor&&) noexcept = default;
PrePostProcessor::~PrePostProcessor() = default;

InputInfo& PrePostProcessor::input() {
    OPENVINO_ASSERT(m_impl->m_inputs.size() == 1,
                    "PrePostProcessor::input() - Model must have exactly one input, got ",
                    m_impl->m_inputs.size());
    return m_impl->m_inputs.front();
}

InputInfo& PrePostProcessor::input(size_t input_index) {
    OPENVINO_ASSERT(m_impl->m_inputs.size() > input_index,
                    "PrePostProcessor::input(size_t) - Model doesn't have input with index ",
                    input_index,
                    ". Total number of inputs is ",
                    m_impl->m_inputs.size());
    return m_impl->m_inputs[input_index];
}

InputInfo& PrePostProcessor::input(const std::string& tensor_name) {
    return m_impl->find_input(tensor_name);
}

OutputInfo& PrePostProcessor::output() {
    OPENVINO_ASSERT(m_impl->m_outputs.size() == 1,
                    "PrePostProcessor::output() - Model must have exactly one output, got ",
                    m_impl->m_outputs.size());
    return m_impl->m_outputs.front();
}

OutputInfo& PrePostProcessor::output(size_t output_index) {
    OPENVINO_ASSERT(m_impl->m_outputs.size() > output_index,
                    "PrePostProcessor::output(size_t) - Model doesn't have input with index ",
                    output_index,
                    ". Total number of inputs is ",
                    m_impl->m_inputs.size());
    return m_impl->m_outputs[output_index];
}

OutputInfo& PrePostProcessor::output(const std::string& tensor_name) {
    return m_impl->find_output(tensor_name);
}

std::ostream& operator<<(std::ostream& str, const PrePostProcessor& prePostProcessor) {
    try {
        prePostProcessor.dump(str);
    } catch (ov::AssertFailure& ex) {
        str << std::endl << "Error occurred: " << ex.what();
    }
    return str;
}

void PrePostProcessor::dump(std::ostream& str) const {
    auto model = m_impl->m_function;
    std::tuple<std::unordered_set<std::string>, bool> existing_names{std::unordered_set<std::string>{}, false};
    for (const auto& input_info : m_impl->m_inputs) {
        input_info.m_impl->dump(str, model, existing_names);
    }
    for (const auto& output_info : m_impl->m_outputs) {
        output_info.m_impl->dump(str);
    }
}

std::shared_ptr<Model> PrePostProcessor::build() {
    auto function = m_impl->m_function;
    std::tuple<std::unordered_set<std::string>, bool> existing_names{std::unordered_set<std::string>{}, false};
    FunctionGuard guard(function);
    bool need_validate = false;
    auto results = function->get_results();
    auto parameters_list = std::list<std::shared_ptr<opset8::Parameter>>(function->get_parameters().begin(),
                                                                         function->get_parameters().end());

    for (const auto& input_info : m_impl->m_inputs) {
        if (input_info.m_impl->build(function, existing_names, parameters_list)) {
            need_validate = true;
        }
    }

    // Add parameters with right order
    {
        while (!function->get_parameters().empty()) {
            function->remove_parameter(*function->get_parameters().begin());
        }
        auto parameters_vec = ParameterVector(parameters_list.begin(), parameters_list.end());
        function->add_parameters(parameters_vec);
    }
    // Validate nodes after preprocessing if needed (no need to repeat it after post-processing)
    if (need_validate) {
        function->validate_nodes_and_infer_types();
    }

    // Post processing
    for (const auto& output_info : m_impl->m_outputs) {
        output_info.m_impl->build(results);
    }
    // Add results with right order
    while (!function->get_results().empty())
        function->remove_result(*function->get_results().begin());
    function->add_results(results);

    // After switching from ModelOptimizer to OVC, the order of
    // applying PrePostProcessing and MOCTransformations has changed:
    //
    // MO path : [fw model conversion -> PrePostProcessing -> MOC] -> nncf
    // OVC path: [fw model conversion -> MOC] -> PrePostProcessing -> nncf
    //
    // Since nncf is applied to a not fully optimized model, extra FQ ops might appear,
    // which can affect both accuracy and performance.
    // PrePostProcessing is not part of OVC, so we have to insert an additional
    // Transformation calls inside PrePostProcessing.
    transformation_pipeline(function);

    guard.reset();
    return function;
}

// ------------------ TensorInfoMemoryType ----------------
TensorInfoMemoryType::~TensorInfoMemoryType() = default;

// --------------------- InputTensorInfo ------------------
InputTensorInfo::InputTensorInfo() : m_impl(std::unique_ptr<InputTensorInfoImpl>(new InputTensorInfoImpl())) {}
InputTensorInfo::~InputTensorInfo() = default;

InputTensorInfo& InputTensorInfo::set_element_type(const element::Type& type) {
    m_impl->set_element_type(type);
    return *this;
}

InputTensorInfo& InputTensorInfo::set_layout(const Layout& layout) {
    m_impl->set_layout(layout);
    return *this;
}

InputTensorInfo& InputTensorInfo::set_spatial_dynamic_shape() {
    m_impl->set_spatial_dynamic_shape();
    return *this;
}

InputTensorInfo& InputTensorInfo::set_spatial_static_shape(size_t height, size_t width) {
    m_impl->set_spatial_static_shape(height, width);
    return *this;
}

// --------------------- InputModelInfo ------------------
InputModelInfo::InputModelInfo() : m_impl(std::unique_ptr<InputModelInfoImpl>(new InputModelInfoImpl())) {}
InputModelInfo::~InputModelInfo() = default;

InputModelInfo& InputModelInfo::set_layout(const Layout& layout) {
    m_impl->set_layout(layout);
    return *this;
}

InputTensorInfo& InputTensorInfo::set_color_format(const ov::preprocess::ColorFormat& format,
                                                   const std::vector<std::string>& sub_names) {
    m_impl->set_color_format(format, sub_names);
    return *this;
}

InputTensorInfo& InputTensorInfo::set_memory_type(const std::string& memory_type) {
    m_impl->set_memory_type(memory_type);
    return *this;
}

InputTensorInfo& InputTensorInfo::set_shape(const PartialShape& shape) {
    m_impl->set_shape(shape);
    return *this;
}

InputTensorInfo& InputTensorInfo::set_from(const ov::Tensor& runtime_tensor) {
    m_impl->set_from(runtime_tensor);
    return *this;
}

// --------------------- PreProcessSteps ------------------

PreProcessSteps::PreProcessSteps() : m_impl(std::unique_ptr<PreProcessStepsImpl>(new PreProcessStepsImpl())) {}
PreProcessSteps::~PreProcessSteps() = default;

PreProcessSteps& PreProcessSteps::clamp(double min_value, double max_value) {
    m_impl->add_clamp(min_value, max_value);
    return *this;
}

PreProcessSteps& PreProcessSteps::scale(float value) {
    m_impl->add_scale_impl(std::vector<float>{value});
    return *this;
}

PreProcessSteps& PreProcessSteps::scale(const std::vector<float>& values) {
    m_impl->add_scale_impl(values);
    return *this;
}

PreProcessSteps& PreProcessSteps::mean(float value) {
    m_impl->add_mean_impl(std::vector<float>{value});
    return *this;
}

PreProcessSteps& PreProcessSteps::mean(const std::vector<float>& values) {
    m_impl->add_mean_impl(values);
    return *this;
}

PreProcessSteps& PreProcessSteps::pad(const std::vector<int>& pads_begin,
                                      const std::vector<int>& pads_end,
                                      float value,
                                      PaddingMode mode) {
    m_impl->add_pad_impl(pads_begin, pads_end, std::vector<float>{value}, mode);
    return *this;
}

PreProcessSteps& PreProcessSteps::pad(const std::vector<int>& pads_begin,
                                      const std::vector<int>& pads_end,
                                      const std::vector<float>& values,
                                      PaddingMode mode) {
    m_impl->add_pad_impl(pads_begin, pads_end, values, mode);
    return *this;
}

PreProcessSteps& PreProcessSteps::convert_element_type(const element::Type& type) {
    m_impl->add_convert_impl(type);
    return *this;
}

PreProcessSteps& PreProcessSteps::resize(ResizeAlgorithm alg, size_t dst_height, size_t dst_width) {
    OPENVINO_ASSERT(dst_height <= static_cast<size_t>(std::numeric_limits<int>::max()) &&
                        dst_width <= static_cast<size_t>(std::numeric_limits<int>::max()),
                    "Resize: Width/Height dimensions cannot be greater than ",
                    std::to_string(std::numeric_limits<int>::max()));
    m_impl->add_resize_impl(alg, static_cast<int>(dst_height), static_cast<int>(dst_width));
    return *this;
}

PreProcessSteps& PreProcessSteps::resize(ResizeAlgorithm alg) {
    m_impl->add_resize_impl(alg, -1, -1);
    return *this;
}

PreProcessSteps& PreProcessSteps::crop(const std::vector<int>& begin, const std::vector<int>& end) {
    m_impl->add_crop_impl(begin, end);
    return *this;
}

PreProcessSteps& PreProcessSteps::convert_layout(const Layout& dst_layout) {
    m_impl->add_convert_layout_impl(dst_layout);
    return *this;
}

PreProcessSteps& PreProcessSteps::convert_layout(const std::vector<uint64_t>& dims) {
    m_impl->add_convert_layout_impl(dims);
    return *this;
}

PreProcessSteps& PreProcessSteps::convert_color(const ov::preprocess::ColorFormat& dst_format) {
    m_impl->add_convert_color_impl(dst_format);
    return *this;
}

PreProcessSteps& PreProcessSteps::custom(const CustomPreprocessOp& preprocess_cb) {
    // 'true' indicates that custom preprocessing step will trigger validate_and_infer_types
    m_impl->actions().emplace_back(
        [preprocess_cb](const std::vector<Output<Node>>& nodes,
                        const std::shared_ptr<ov::Model>&,
                        PreprocessingContext&) {
            OPENVINO_ASSERT(nodes.size() == 1,
                            "Can't apply custom preprocessing step for multi-plane input. Suggesting to convert "
                            "current image to RGB/BGR color format using 'convert_color'");
            return std::make_tuple(std::vector<Output<Node>>{preprocess_cb(nodes[0])}, true);
        },
        "custom");
    return *this;
}

PreProcessSteps& PreProcessSteps::reverse_channels() {
    m_impl->add_reverse_channels();
    return *this;
}

// --------------------- OutputTensorInfo ------------------
OutputTensorInfo::OutputTensorInfo() : m_impl(std::unique_ptr<OutputTensorInfoImpl>(new OutputTensorInfoImpl())) {}
OutputTensorInfo::~OutputTensorInfo() = default;

OutputTensorInfo& OutputTensorInfo::set_element_type(const element::Type& type) {
    m_impl->set_element_type(type);
    return *this;
}

OutputTensorInfo& OutputTensorInfo::set_layout(const Layout& layout) {
    m_impl->set_layout(layout);
    return *this;
}

// --------------------- OutputModelInfo ------------------
OutputModelInfo::OutputModelInfo() : m_impl(std::unique_ptr<OutputModelInfoImpl>(new OutputModelInfoImpl())) {}
OutputModelInfo::~OutputModelInfo() = default;

OutputModelInfo& OutputModelInfo::set_layout(const Layout& layout) {
    m_impl->set_layout(layout);
    return *this;
}

OutputModelInfo& OutputModelInfo::set_color_format(const ov::preprocess::ColorFormat& format,
                                                   const std::vector<std::string>& sub_names) {
    m_impl->set_color_format(format);
    return *this;
}

// --------------------- PostProcessSteps ------------------

PostProcessSteps::PostProcessSteps() : m_impl(std::unique_ptr<PostProcessStepsImpl>(new PostProcessStepsImpl())) {}
PostProcessSteps::~PostProcessSteps() = default;

PostProcessSteps& PostProcessSteps::clamp(double min_value, double max_value) {
    m_impl->add_clamp(min_value, max_value);
    return *this;
}

PostProcessSteps& PostProcessSteps::convert_element_type(const element::Type& type) {
    m_impl->add_convert_impl(type);
    return *this;
}

PostProcessSteps& PostProcessSteps::convert_layout(const Layout& dst_layout) {
    m_impl->add_convert_layout_impl(dst_layout);
    return *this;
}

PostProcessSteps& PostProcessSteps::convert_layout(const std::vector<uint64_t>& dims) {
    m_impl->add_convert_layout_impl(dims);
    return *this;
}

PostProcessSteps& PostProcessSteps::convert_color(const ov::preprocess::ColorFormat& dst_format) {
    m_impl->add_convert_color_impl(dst_format);
    return *this;
}

PostProcessSteps& PostProcessSteps::custom(const CustomPostprocessOp& postprocess_cb) {
    // 'true' indicates that custom postprocessing step will trigger validate_and_infer_types
    m_impl->actions().emplace_back(
        [postprocess_cb](const Output<ov::Node>& node, PostprocessingContext&) {
            return std::make_tuple(postprocess_cb(node), true);
        },
        "custom");
    return *this;
}

}  // namespace preprocess
}  // namespace ov
