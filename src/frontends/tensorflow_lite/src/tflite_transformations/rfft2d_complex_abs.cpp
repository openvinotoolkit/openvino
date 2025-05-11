// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tflite_transformations/rfft2d_complex_abs.h"

#include <memory>

#include "openvino/core/rt_info.hpp"
#include "openvino/opsets/opset9.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "tflite_ops/complex_abs.h"
#include "tflite_ops/rfft2d.h"
#include "utils.hpp"

using namespace std;
using namespace ov::pass;
using namespace ov::pass::pattern;
using namespace ov::opset9;
using namespace ov::frontend::tensorflow_lite;

pass::Rfft2dSimplifier::Rfft2dSimplifier() {
    auto rfft2d_label = wrap_type<tensorflow_lite::Rfft2d>();
    auto reshape_label = wrap_type<Reshape>({rfft2d_label, pattern::any_input()});
    auto complex_abs_label = wrap_type<tensorflow_lite::ComplexAbs>({reshape_label});

    matcher_pass_callback callback = [=](Matcher& m) {
        auto pattern_map = m.get_pattern_map();
        auto rfft_original_node = pattern_map.at(rfft2d_label);
        auto reshape_original_node = pattern_map.at(reshape_label);
        auto abs_original_node = pattern_map.at(complex_abs_label);

        auto rfft = make_shared<RDFT>(rfft_original_node->get_input_source_output(0),
                                      Constant::create(element::i64, {2}, {-2, -1}),
                                      rfft_original_node->get_input_source_output(1));
        auto split = make_shared<Split>(rfft, Constant::create(element::i64, {}, {-1}), 2);

        auto real = make_shared<Unsqueeze>(split->output(0), Constant::create(element::i64, {}, {-1}));
        auto imag = make_shared<Unsqueeze>(split->output(1), Constant::create(element::i64, {}, {-1}));

        auto reshape_real =
            reshape_original_node->clone_with_new_inputs({real, reshape_original_node->get_input_source_output(1)});
        auto reshape_imag =
            reshape_original_node->clone_with_new_inputs({imag, reshape_original_node->get_input_source_output(1)});

        auto two = make_shared<ConvertLike>(Constant::create(element::i64, {}, {2}), reshape_real);
        auto complex_abs = make_shared<Sqrt>(
            make_shared<Add>(make_shared<Power>(reshape_real, two), make_shared<Power>(reshape_imag, two)));

        complex_abs->output(0).set_names(abs_original_node->output(0).get_names());
        abs_original_node->output(0).replace(complex_abs->output(0));
        complex_abs->set_friendly_name(abs_original_node->get_friendly_name());
        ov::copy_runtime_info({rfft_original_node, reshape_original_node, abs_original_node}, complex_abs);
        return true;
    };

    auto m =
        std::make_shared<pattern::Matcher>(complex_abs_label, "ov::frontend::tensorflow_lite::pass::Rfft2dSimplifier");
    register_matcher(m, callback);
}
