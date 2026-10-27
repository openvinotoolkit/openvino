// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/plugin/simple_math.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/pass/serialize.hpp"
#include <algorithm>
#include <sstream>
#include <vector>
#include <string>

namespace cldnn {

/// @brief This primitive executes a custom kernel provided by the application
/// @details The application is required to provide all relevant details for executing the custom kernel
/// such as: sources, entry point, work sizes and parameter bindings.
struct custom_gpu_primitive : public primitive_base<custom_gpu_primitive> {
    CLDNN_DECLARE_PRIMITIVE(custom_gpu_primitive)

    custom_gpu_primitive() : primitive_base("", {}) {}

    /// @brief Custom primitive kernel argument type
    enum arg_type {
        arg_input,
        arg_output,
        arg_internal,
    };
    //
    /// @brief Custom primitive kernel argument index
    using arg_index = uint32_t;
    //
    /// @brief Custom primitive kernel argument description
    struct arg_desc {
        arg_type type;
        arg_index index;
        std::string size_expr;

        bool operator==(const arg_desc& rhs) const {
            return (type == rhs.type && index == rhs.index && size_expr == rhs.size_expr);
        }

        void save(BinaryOutputBuffer& ob) const {
            ob << make_data(&type, sizeof(arg_type));
            ob << index;
            ob << size_expr;
        }

        void load(BinaryInputBuffer& ib) {
            ib >> make_data(&type, sizeof(arg_type));
            ib >> index;
            ib >> size_expr;
        }
    };

    static void update_work_group_size(const ov::PartialShape& dims,
                                       int calcWgDimInputIdx,
                                       const ov::PartialShape& inputDims,
                                       const std::vector<std::string>& globalSizeRules,
                                       const std::vector<std::string>& localSizeRules,
                                       std::vector<size_t>& gws,
                                       std::vector<size_t>& lws) {
#define GetDim(DIM) static_cast<int>(DIM.is_dynamic() ? -1 : DIM.get_length())

        gws.clear();
        lws.clear();

        int batchDim = 0, featureDim = 0, yDim = 0, xDim = 0;
        // if calcWgDimInputIdx is greater than -1, take dimension from input
        if (calcWgDimInputIdx >= 0) {
            xDim = GetDim(inputDims[inputDims.size() - 1]);
            yDim = dims.size() > 1 ? GetDim(inputDims[inputDims.size() - 2]) : 0;
            featureDim = dims.size() > 2 ? GetDim(inputDims[inputDims.size() - 3]) : 0;
            batchDim = dims.size() > 3 ? GetDim(inputDims[inputDims.size() - 4]) : 0;
        } else {
            batchDim = (dims.size() > 0) ? GetDim(dims[0]) : 1;
            featureDim = (dims.size() > 1) ? GetDim(dims[1]) : 1;
            yDim = (dims.size() > 2) ? GetDim(dims[2]) : 1;
            xDim = (dims.size() > 3) ? GetDim(dims[3]) : 1;
        }
        const std::map<char, int> vars = {
            {'b', batchDim},    {'B', batchDim},
            {'f', featureDim},  {'F', featureDim},
            {'y', yDim},        {'Y', yDim},
            {'x', xDim},        {'X', xDim},
        };
        for (const auto& rule : globalSizeRules) {
            SimpleMathExpression expr;
            expr.SetVariables(vars);
            expr.SetExpression(rule);
            gws.push_back(expr.Evaluate());
        }
        for (const auto& rule : localSizeRules) {
            SimpleMathExpression expr;
            expr.SetVariables(vars);
            expr.SetExpression(rule);
            lws.push_back(expr.Evaluate());
        }
    }

    /// @brief Constructs custom_gpu_primitive primitive
    /// @param id This primitive id.
    /// @param input Input primitive ids.
    /// @param kernels_code Source code for the kernel
    /// @param kernel_entry_point The name of the entry point function in the kernel
    /// @param kernel_arguments Argument bindings for the entry point function
    /// @param build_options Build options/flags used during the compilation of the custom kernel
    /// @param output_layout Output layout declared by the primitive
    /// @param gws Global work sizes
    /// @param lws Local work sizes
    custom_gpu_primitive(const primitive_id& id,
                         const std::vector<input_info>& inputs,
                         const std::vector<std::string>& kernels_code,
                         const std::string& kernel_entry_point,
                         const std::vector<arg_desc>& kernel_arguments,
                         const std::string& build_options,
                         const std::vector<layout>& output_layouts,
                         const std::vector<size_t>& gws = {},
                         const std::vector<size_t>& lws = {},
                         const std::shared_ptr<ov::Node>& op = nullptr,
                         const int calcWgDimInputIdx = -1,
                         const std::vector<std::string> globalSizeRules = {},
                         const std::vector<std::string> localSizeRules = {})
        : primitive_base(id, inputs, output_layouts.size(), {optional_data_type()}, {output_layouts[0].data_padding}),
          kernel_entry_point(kernel_entry_point),
          kernel_arguments(kernel_arguments),
          build_options(build_options),
          output_layouts(output_layouts),
          gws(!gws.empty() ? gws : std::vector<size_t>{output_layouts[0].count()}),
          lws(lws),
          kernels_code(kernels_code),
          op(op),
          calcWgDimInputIdx(calcWgDimInputIdx),
          globalSizeRules(globalSizeRules),
          localSizeRules(localSizeRules) {}

    /// @brief The name of the entry point function in the kernel
    const std::string kernel_entry_point;
    /// @brief Argument bindings for the entry point function
    const std::vector<arg_desc> kernel_arguments;
    /// @brief The kernel's build options
    const std::string build_options;
    /// @brief The output layout declared by the primitive
    const std::vector<layout> output_layouts;
    /// @brief The global working sizes
    const std::vector<size_t> gws;
    /// @brief The local working sizes
    const std::vector<size_t> lws;
    /// @brief Source code for the kernel
    const primitive_id_arr kernels_code;
    /// @brief Original IR op
    std::shared_ptr<ov::Node> op;
    /// @brief -1: mean calc gws via output, else calc gws via inputs
    const int calcWgDimInputIdx = -1;
    /// @brief Custom provided rules for calc work sizes.
    const std::vector<std::string> globalSizeRules;
    const std::vector<std::string> localSizeRules;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, kernel_entry_point);
        for (const auto& args : kernel_arguments) {
            seed = hash_combine(seed, args.index);
            seed = hash_combine(seed, args.type);
            seed = hash_combine(seed, args.size_expr);
        }
        seed = hash_combine(seed, build_options);
        seed = hash_range(seed, kernels_code.begin(), kernels_code.end());
        seed = hash_range(seed, gws.begin(), gws.end());
        seed = hash_range(seed, lws.begin(), lws.end());
        seed = hash_combine(seed, calcWgDimInputIdx);
        seed = hash_range(seed, globalSizeRules.begin(), globalSizeRules.end());
        seed = hash_range(seed, localSizeRules.begin(), localSizeRules.end());
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs)) {
            return false;
        }

        auto rhs_casted = downcast<const custom_gpu_primitive>(rhs);

        if (kernel_entry_point != rhs_casted.kernel_entry_point) {
            return false;
        }

        if (build_options != rhs_casted.build_options) {
            return false;
        }

        if (kernel_arguments != rhs_casted.kernel_arguments) {
            return false;
        }

        if (kernels_code != rhs_casted.kernels_code) {
            return false;
        }

        if (gws != rhs_casted.gws) {
            return false;
        }

        return lws == rhs_casted.lws && calcWgDimInputIdx == rhs_casted.calcWgDimInputIdx &&
               globalSizeRules == rhs_casted.globalSizeRules && localSizeRules == rhs_casted.localSizeRules;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<custom_gpu_primitive>::save(ob);
        ob << kernel_entry_point;
        ob << kernel_arguments;
        ob << build_options;
        ob << output_layouts;
        ob << gws;
        ob << lws;
        ob << kernels_code;
        const uint64_t schema = 0x435553544f4d3032ULL;
        ob << schema;
        ob << calcWgDimInputIdx;
        ob << globalSizeRules;
        ob << localSizeRules;
        std::string shape_model;
        const bool dynamic_output = std::any_of(output_layouts.begin(), output_layouts.end(),
                                               [](const layout& output) { return output.is_dynamic(); });
        if (dynamic_output) {
            OPENVINO_ASSERT(op, "[GPU] Dynamic custom primitive has no shape inference operation");
            ov::ParameterVector parameters;
            ov::OutputVector inputs;
            for (size_t port = 0; port < op->get_input_size(); ++port) {
                auto parameter = std::make_shared<ov::op::v0::Parameter>(
                    op->get_input_element_type(port), op->get_input_partial_shape(port));
                parameters.push_back(parameter);
                inputs.push_back(parameter);
            }
            auto independent = op->clone_with_new_inputs(inputs);
            independent->validate_and_infer_types();
            auto model = std::make_shared<ov::Model>(independent->outputs(), parameters);
            std::ostringstream xml, weights;
            ov::pass::Serialize(xml, weights).run_on_model(model);
            OPENVINO_ASSERT(weights.str().empty(), "[GPU] Custom shape inference model must not contain weights");
            shape_model = xml.str();
        }
        ob << shape_model;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<custom_gpu_primitive>::load(ib);
        ib >> *const_cast<std::string*>(&kernel_entry_point);
        ib >> *const_cast<std::vector<arg_desc>*>(&kernel_arguments);
        ib >> *const_cast<std::string*>(&build_options);
        ib >> *const_cast<std::vector<layout>*>(&output_layouts);
        ib >> *const_cast<std::vector<size_t>*>(&gws);
        ib >> *const_cast<std::vector<size_t>*>(&lws);
        ib >> *const_cast<primitive_id_arr*>(&kernels_code);
        uint64_t schema = 0;
        ib >> schema;
        OPENVINO_ASSERT(schema == 0x435553544f4d3032ULL,
                        "[GPU] Incompatible custom primitive cache schema");
        ib >> *const_cast<int*>(&calcWgDimInputIdx);
        ib >> *const_cast<std::vector<std::string>*>(&globalSizeRules);
        ib >> *const_cast<std::vector<std::string>*>(&localSizeRules);
        std::string shape_model;
        ib >> shape_model;
        const bool dynamic_output = std::any_of(output_layouts.begin(), output_layouts.end(),
                                               [](const layout& output) { return output.is_dynamic(); });
        OPENVINO_ASSERT(dynamic_output == !shape_model.empty(),
                        "[GPU] Custom shape inference model does not match the cached output contract");
        op.reset();
        if (dynamic_output) {
            auto model = ib.read_model(shape_model);
            OPENVINO_ASSERT(model && model->get_parameters().size() == input.size() &&
                                model->get_results().size() == output_layouts.size(),
                            "[GPU] Custom shape inference model arity differs");
            auto restored = model->get_results().front()->input_value(0).get_node_shared_ptr();
            OPENVINO_ASSERT(restored->get_input_size() == input.size() &&
                                restored->get_output_size() == output_layouts.size(),
                            "[GPU] Custom shape inference operation arity differs");
            for (size_t port = 0; port < input.size(); ++port) {
                OPENVINO_ASSERT(restored->input_value(port) == model->get_parameters()[port]->output(0),
                                "[GPU] Custom shape inference input order differs");
            }
            for (size_t port = 0; port < output_layouts.size(); ++port) {
                OPENVINO_ASSERT(model->get_results()[port]->input_value(0) == restored->output(port) &&
                                    element_type_to_data_type(restored->get_output_element_type(port)) == output_layouts[port].data_type &&
                                    restored->get_output_partial_shape(port).compatible(output_layouts[port].get_partial_shape()),
                                "[GPU] Custom shape inference output contract differs");
            }
            // Retain the shape model, including any extension-library ownership it carries.
            op = std::shared_ptr<ov::Node>(model, restored.get());
        }
    }
};
}  // namespace cldnn
