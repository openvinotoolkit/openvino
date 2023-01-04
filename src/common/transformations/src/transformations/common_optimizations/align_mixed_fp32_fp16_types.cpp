//
// Created by pesir on 10/18/2022.
//

#include "transformations/common_optimizations/align_mixed_fp32_fp16_types.hpp"

#include <openvino/opsets/opset1.hpp>
#include <openvino/opsets/opset8.hpp>
#include <ngraph/rt_info.hpp>
#include "itt.hpp"
#include "openvino/op/util/precision_sensitive_attribute.hpp"
#include "transformations/convert_precision.hpp"
#include "transformations/rt_info/decompression.hpp"
#include "transformations/rt_info/disable_fp16_compression.hpp"
#include "transformations/convert_precision.hpp"

using namespace ov;

bool ov::pass::AlignMixedFP32FP16Types::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(AlignMixedFP32FP16Types);

    std::function<bool(const std::shared_ptr<Node>&)> insert_converts_before = [&](const std::shared_ptr<Node>& node) {
        bool is_changed = false;
        for (const auto& input : node->inputs()) {
            const auto& incoming_output = input.get_source_output();
            const auto& incoming_node = incoming_output.get_node_shared_ptr();

            if (fp16_compression_is_disabled(incoming_node))
                continue;  // we are in the middle

            if (!incoming_output.get_element_type().is_real())
                continue;

            const auto& existing_convert = std::dynamic_pointer_cast<opset8::Convert>(incoming_node);
            if (existing_convert) {
                existing_convert->set_convert_element_type(element::f32);
            } else {
                auto convert = std::make_shared<opset8::Convert>(incoming_output, ov::element::f32);
                convert->set_friendly_name(incoming_node->get_friendly_name() +
                                           "_decompressed_to_f32");
                copy_runtime_info(incoming_node, convert);
                input.replace_source_output(convert);
                ov::disable_fp16_compression(convert);
                ov::mark_as_decompression(convert);
            }
            is_changed = true;
        }
        return is_changed;
    };

    std::function<bool(const std::shared_ptr<Node>&)> insert_converts_after = [&](const std::shared_ptr<Node>& node) {
        bool is_changed = false;
        for (const auto& output : node->outputs()) {
            for (const auto& out_inputs : output.get_target_inputs()) {
                auto out_node = out_inputs.get_node()->shared_from_this();
                if (fp16_compression_is_disabled(out_node) || is_precision_sensitive(out_inputs))
                    continue;
                if (!out_inputs.get_element_type().is_real())
                    continue;

                // todo: if we don't skip Results there is an error on GPU at
                // src/plugins/intel_gpu/src/plugin/ops/result.cpp:33
                // IE_THROW() << "Can't find output " << inputID << " in OutputsDataMap";
                if (std::dynamic_pointer_cast<opset8::Result>(out_node))
                    continue;

                const auto& existing_convert = std::dynamic_pointer_cast<opset8::Convert>(out_node);

                if (existing_convert) {
                    existing_convert->set_convert_element_type(element::f32);
                } else {
                    // element_type of this convert will be changed automatically to f16 after
                    // ConvertPrecision(f32 -> f16). It's kept here f32 to keep ov::Model validatable
                    auto convert = std::make_shared<opset8::Convert>(output, element::f32);
                    copy_runtime_info(node, convert);
                    convert->set_friendly_name(node->get_friendly_name() + "_compressed_to_f16");
                    out_inputs.replace_source_output(convert);
                }
                is_changed = true;
            }
        }
        return is_changed;
    };


    bool is_changed = false;
    for (auto& node : model->get_ordered_ops()) {
        if (!fp16_compression_is_disabled(node))
            continue;

        is_changed |= insert_converts_before(node);
        is_changed |= insert_converts_after(node);
    }

    return true;
}
