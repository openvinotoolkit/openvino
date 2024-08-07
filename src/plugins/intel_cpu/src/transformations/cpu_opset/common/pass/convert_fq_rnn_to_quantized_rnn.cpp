// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_fq_rnn_to_quantized_rnn.hpp"

#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gru_sequence.hpp"
#include "openvino/op/lstm_sequence.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/core/rt_info.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "transformations/rt_info/disable_constant_folding.hpp"

#include "itt.hpp"

ov::intel_cpu::ConvertFqRnnToQuantizedRnn::ConvertFqRnnToQuantizedRnn() {
    MATCHER_SCOPE(ConvertFqRnnToQuantizedRnn);

    auto X_m = ov::pass::pattern::any_input();
    auto convert_X = pass::pattern::wrap_type<op::v0::Convert>({X_m});
    auto input_shift_X = pass::pattern::wrap_type<op::v0::Constant>();
    auto subtract_X = pass::pattern::wrap_type<op::v1::Subtract>({convert_X, input_shift_X});
    auto input_scale_X = pass::pattern::wrap_type<op::v0::Constant>();

    auto deq_X = std::make_shared<pass::pattern::op::Or>(
        OutputVector{
            pass::pattern::wrap_type<op::v1::Multiply>({convert_X, input_scale_X}),
            pass::pattern::wrap_type<op::v1::Multiply>({subtract_X, input_scale_X}),
        });

    auto H_m = ov::pass::pattern::any_input();
    auto convert_H = pass::pattern::wrap_type<op::v0::Convert>({H_m});
    auto input_shift_H = pass::pattern::wrap_type<op::v0::Constant>();
    auto subtract_H = pass::pattern::wrap_type<op::v1::Subtract>({convert_H, input_shift_H});
    auto input_scale_H = pass::pattern::wrap_type<op::v0::Constant>();

    auto deq_H = std::make_shared<pass::pattern::op::Or>(
        OutputVector{
            pass::pattern::wrap_type<op::v1::Multiply>({convert_H, input_scale_H}),
            pass::pattern::wrap_type<op::v1::Multiply>({subtract_H, input_scale_H}),
        });

    auto H_as_const = ov::pass::pattern::wrap_type<op::v0::Constant>();
    auto H_in = std::make_shared<ov::pass::pattern::op::Or>(
        OutputVector {
            deq_H,
            H_as_const
        });

    auto cell_state_m = ov::pass::pattern::any_input(); // for LSTM
    auto sequence_length_m = ov::pass::pattern::any_input(); // for Sequences

    auto W_m = ov::pass::pattern::wrap_type<op::v0::Constant>();
    auto convert_W = ov::pass::pattern::wrap_type<op::v0::Convert>({W_m});
    auto weights_scale_W = ov::pass::pattern::wrap_type<op::v0::Constant>();
    auto deq_W = ov::pass::pattern::wrap_type<op::v1::Multiply>({convert_W, weights_scale_W});

    auto R_m = ov::pass::pattern::wrap_type<op::v0::Constant>();
    auto convert_R = ov::pass::pattern::wrap_type<op::v0::Convert>({R_m});
    auto weights_scale_R = ov::pass::pattern::wrap_type<op::v0::Constant>();
    auto deq_R = ov::pass::pattern::wrap_type<op::v1::Multiply>({convert_R, weights_scale_R});

    const auto B_m = ov::pass::pattern::wrap_type<op::v0::Constant>();

    auto lstm_seq_m = ov::pass::pattern::wrap_type<op::v5::LSTMSequence>({deq_X, H_in, cell_state_m, sequence_length_m, deq_W, deq_R, B_m});
    auto gru_seq_m  = ov::pass::pattern::wrap_type<op::v5::GRUSequence> ({deq_X, H_in,               sequence_length_m, deq_W, deq_R, B_m});

    auto rnn_pattern = std::make_shared<ov::pass::pattern::op::Or>(
        OutputVector {
            lstm_seq_m,
            gru_seq_m
        });

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        auto rnn = m.get_match_root();
        if (!rnn || transformation_callback(rnn))
            return false;

        const auto& pattern_map  = m.get_pattern_value_map();
        const auto& activation   = pattern_map.at(X_m);
        const auto  hidden_state_it = pattern_map.find(H_m);

        ov::Output<ov::Node> hidden_state;
        if (hidden_state_it != pattern_map.end()) { // is it H(i8/u8) -> dequantized -> RNN pattern?
            hidden_state = hidden_state_it->second;
        } else {
            hidden_state = pattern_map.at(H_as_const); // if not, then it is just H (f32 const) -> RNN
        }

        const auto& weights      = pattern_map.at(W_m);
        const auto& r_weights    = pattern_map.at(R_m);
        const auto& bias         = pattern_map.at(B_m);

        // At the moment there is no optimal primitive for i8 LSTM in the plugin,
        // thus it's better to leave such cases in f32.
        if (hidden_state.get_element_type() == element::i8) {
            const auto& w_convert = pattern_map.at(convert_W);
            const auto& w_mul = pattern_map.at(deq_W);
            const auto& r_convert = pattern_map.at(convert_R);
            const auto& r_mul = pattern_map.at(deq_R);

            ov::enable_constant_folding(w_convert.get_node_shared_ptr());
            ov::enable_constant_folding(w_mul.get_node_shared_ptr());
            ov::enable_constant_folding(r_convert.get_node_shared_ptr());
            ov::enable_constant_folding(r_mul.get_node_shared_ptr());

            return false;
        }

        std::shared_ptr<ov::Node> rnn_quantized;

        if (const auto lstm_seq = ov::as_type_ptr<op::v5::LSTMSequence>(rnn)) {
            const auto& cell_state = pattern_map.at(cell_state_m);
            const auto& sequence_length = pattern_map.at(sequence_length_m);

            // @todo prototype removal of unnecessary fq between two consecutive rnn nodes
            auto rnn_quantized_tr = std::make_shared<op::TypeRelaxed<op::v5::LSTMSequence>>(
                element::TypeVector{ element::f32, element::f32, element::f32, element::f32, element::f32, element::f32, element::f32 },
                element::TypeVector{ element::f32, element::f32, element::f32 },
                op::TemporaryReplaceOutputType(activation, element::f32).get(),
                op::TemporaryReplaceOutputType(hidden_state, element::f32).get(),
                op::TemporaryReplaceOutputType(cell_state, element::f32).get(),
                op::TemporaryReplaceOutputType(sequence_length, element::f32).get(),
                op::TemporaryReplaceOutputType(weights, element::f32).get(),
                op::TemporaryReplaceOutputType(r_weights, element::f32).get(),
                op::TemporaryReplaceOutputType(bias, element::f32).get(),
                lstm_seq->get_hidden_size(),
                lstm_seq->get_direction(),
                lstm_seq->get_activations_alpha(),
                lstm_seq->get_activations_beta(),
                lstm_seq->get_activations(),
                lstm_seq->get_clip());

            rnn_quantized_tr->set_overridden_output_type(hidden_state.get_element_type(), 1);
            rnn_quantized = rnn_quantized_tr;
        } else if (const auto gru_seq = ov::as_type_ptr<op::v5::GRUSequence>(rnn)) {
            const auto& sequence_length = pattern_map.at(sequence_length_m);

            auto rnn_quantized_tr = std::make_shared<op::TypeRelaxed<op::v5::GRUSequence>>(
                std::vector<ov::element::Type>{ element::f32, element::f32, element::f32, element::f32, element::f32, element::f32},
                std::vector<ov::element::Type>{ element::f32, element::f32 },
                op::TemporaryReplaceOutputType(activation, element::f32).get(),
                op::TemporaryReplaceOutputType(hidden_state, element::f32).get(),
                op::TemporaryReplaceOutputType(sequence_length, element::f32).get(),
                op::TemporaryReplaceOutputType(weights, element::f32).get(),
                op::TemporaryReplaceOutputType(r_weights, element::f32).get(),
                op::TemporaryReplaceOutputType(bias, element::f32).get(),
                gru_seq->get_hidden_size(),
                gru_seq->get_direction(),
                gru_seq->get_activations(),
                gru_seq->get_activations_alpha(),
                gru_seq->get_activations_beta(),
                gru_seq->get_clip(),
                gru_seq->get_linear_before_reset());

            rnn_quantized_tr->set_overridden_output_type(hidden_state.get_element_type(), 1);
            rnn_quantized = rnn_quantized_tr;
        } else {
            return false;
        }

        // input scales (Multiply per tensor) and weights_scales (Multiply per multiple dimensions) must be present
        const auto& input_scale_output   = pattern_map.at(input_scale_X);
        const auto& weights_scale_output = pattern_map.at(weights_scale_W);
        // extract constant values
        const auto input_scale_constant   = std::dynamic_pointer_cast<op::v0::Constant>(input_scale_output.get_node_shared_ptr());
        const auto weights_scale_constant = std::dynamic_pointer_cast<op::v0::Constant>(weights_scale_output.get_node_shared_ptr());

        if (!input_scale_constant || !weights_scale_constant)
            return false;

        const float* input_scale_ptr = input_scale_constant->get_data_ptr<float>();
        if (*input_scale_ptr == 0.f)
            OPENVINO_THROW("Cannot handle zero input scale");

        const float input_scale  = 1 / *input_scale_ptr;
        std::vector<float> weights_scales  = weights_scale_constant->get_vector<float>();

        // transform dequantization scales into quantization ones
        std::transform(weights_scales.begin(), weights_scales.end(), weights_scales.begin(), [](float& scale) { return 1 / scale; });

        auto& runtime_info = rnn_quantized->get_rt_info();

        // use runtime information to store input and weight scales
        runtime_info["inputScale"]    = input_scale;
        runtime_info["weightsScales"] = weights_scales;

        // input shift (Subtract) is optional
        const auto input_shift_it = pattern_map.find(input_shift_X);

        if (input_shift_it != pattern_map.end()) {
            const auto  input_shift_constant = std::dynamic_pointer_cast<op::v0::Constant>(input_shift_it->second.get_node_shared_ptr());
            const float* input_shift_ptr      = input_shift_constant->get_data_ptr<float>();
            runtime_info["inputShift"] = *input_shift_ptr;
        }

        auto H_outputs = rnn->output(1).get_target_inputs();
        rnn_quantized->set_friendly_name(rnn->get_friendly_name());
        ov::copy_runtime_info(rnn, rnn_quantized);
        ov::replace_node(rnn, rnn_quantized);

        /* in case of pattern:
         * H(u8,i8) -> dequantize -> RNN
         * dequantize has to be inserted after H output port since
         * oneDNN supports only equal data types on H in/out ports
         * either: u8u8, i8i8 or f32f32 */
        if (hidden_state_it != pattern_map.end()) {
            const auto& convert  = pattern_map.at(convert_H).get_node_shared_ptr();
            const auto  subtract_it = pattern_map.find(subtract_H);
            const auto& multiply = rnn->get_input_node_shared_ptr(1);

            auto new_convert  = convert->clone_with_new_inputs({rnn_quantized->output(1)});
            std::shared_ptr<Node> multiply_input = new_convert;
            // dequantize with subtract
            if (subtract_it != pattern_map.end()) {
                const auto subtract = std::dynamic_pointer_cast<op::v1::Subtract>(subtract_it->second.get_node_shared_ptr());
                multiply_input = subtract->clone_with_new_inputs({multiply_input, subtract->input_value(1)});
            }

            auto new_multiply = multiply->clone_with_new_inputs({multiply_input, multiply->input_value(1)});
            new_multiply->set_friendly_name(rnn_quantized->get_friendly_name() + ".1");

            for (auto output : H_outputs) {
                output.replace_source_output(new_multiply);
            }
        }

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(rnn_pattern, matcher_name);
    this->register_matcher(m, callback);
}
