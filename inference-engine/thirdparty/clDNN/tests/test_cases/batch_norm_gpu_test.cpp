/*
// Copyright (c) 2016 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////
#include <gtest/gtest.h>
#include "api/memory.hpp"
#include <api/input_layout.hpp>
#include "api/batch_norm.hpp"
#include <api/topology.hpp>
#include <api/network.hpp>
#include <api/engine.hpp>
#include "test_utils/test_utils.h"
#include <api/reorder.hpp>
#include <api/data.hpp>
#include <api/mutable_data.hpp>

using namespace cldnn;
using namespace tests;

TEST(batch_normalization_gpu, basic_in2x3x2x2) {
    //  Mean   : 3x2x2
    //  Input  : 2x3x2x2
    //  Output : 2x3x2x2

    //  Input:
    //  f0: b0:  1    2  -10   b1:   0    0     -11
    //  f0: b0:  3    4  -14   b1:   0.5 -0.5   -15  
    //  f1: b0:  5    6  -12   b1:   1.5  5.2   -13     
    //  f1: b0:  7    8  -16   b1:   12   9     -17
    //
    //  Mean
    //  f0: -3.3333
    //  f1: -0.3583
    //
    //  Variance
    //  f0: 44.9305
    //  f1: 107.0624

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 2, 2, 3, 2 } });
    auto mean = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 2, 1, 1 } });
    auto variance = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 2, 1, 1 } });

    float epsilon = 0.0001f;

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(data("mean", mean));
    topology.add(data("variance", variance));
    topology.add(batch_norm("batch_norm", "input", "mean", "variance", epsilon));

    set_values(input, {
        1.f, 0.f, 5.f, 1.5f,
        2.f, 0.f, 6.f, 5.2f,
        -10.f, -11.f, -12.f, -13.f,
        3.f, 0.5f, 7.f, 12.f,
        4.f, -0.5f, 8.f, 9.f,
        -14.f, -15.f, -16.f, -17.f
    });

    set_values(mean, { -3.3333f, -0.3583f });
    set_values(variance, { 44.9305f, 107.0624f });

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("batch_norm").get_memory();
    auto output_ptr = output.pointer<float>();

    for (int j = 0; j < 2; ++j) { //F
        float sum = 0, var = 0;
        for (int i = 0; i < 2; ++i) { //B
            for (int k = 0; k < 2; ++k) { //Y
                for (int l = 0; l < 3; ++l) { //X
                    float data = output_ptr[i + 2*j + 2*2*l + 2*2*3*k];
                    sum += data;
                    var += data * data;
                }
            }
        }
        sum /= 2 * 3 * 2;
        var /= 2 * 3 * 2;

        EXPECT_NEAR(sum, 0, 1e-03F);
        EXPECT_NEAR(var, 1, 1e-03F);
    }
}

TEST(batch_normalization_gpu, basic_in2x3x2x2_scale_shift) {
	//  Mean   : 3x2x2
	//  Input  : 2x3x2x2
	//  Output : 2x3x2x2

	//  Input:
	//  f0: b0:  1    2  -10   b1:   0    0     -11
	//  f0: b0:  3    4  -14   b1:   0.5 -0.5   -15  
	//  f1: b0:  5    6  -12   b1:   1.5  5.2   -13     
	//  f1: b0:  7    8  -16   b1:   12   9     -17
	//
	//  Mean
	//  f0: -3.3333
	//  f1: -0.3583
	//
	//  Variance
	//  f0: 44.9305
	//  f1: 107.0624
	//
	//  Scale
	//  f0: 2.0
	//  f1: 1.0
	//
	//  Shift
	//  f0: 0.0
	//  f1: 5.0

	const auto& engine = get_test_engine();

	auto input = memory::allocate(engine, { data_types::f32, format::yxfb,{ 2, 2, 3, 2 } });
	auto mean = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 2, 1, 1 } });
	auto variance = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 2, 1, 1 } });
	auto scale = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 2, 1, 1 } });
	auto shift = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 2, 1, 1 } });

	float epsilon = 0.0001f;

	topology topology;
	topology.add(input_layout("input", input.get_layout()));
	topology.add(data("mean", mean));
	topology.add(data("variance", variance));
	topology.add(data("scale", scale));
	topology.add(data("shift", shift));
	topology.add(batch_norm("batch_norm", "input", "mean", "variance", "scale", "shift",  epsilon));

	set_values(input, {
		1.f, 0.f, 5.f, 1.5f,
		2.f, 0.f, 6.f, 5.2f,
		-10.f, -11.f, -12.f, -13.f,
		3.f, 0.5f, 7.f, 12.f,
		4.f, -0.5f, 8.f, 9.f,
		-14.f, -15.f, -16.f, -17.f
	});

	set_values(mean, { -3.3333f, -0.3583f });
	set_values(variance, { 44.9305f, 107.0624f });
	set_values(scale, { 2.f, 1.f });
	set_values(shift, { 0.f, 5.f });

	network network(engine, topology);

	network.set_input_data("input", input);

	auto outputs = network.execute();

	auto output = outputs.at("batch_norm").get_memory();
	auto output_ptr = output.pointer<float>();

	for (int j = 0; j < 2; ++j) { //F
		float sum = 0, var = 0;

		auto scalep = scale.pointer<float>();
		auto shiftp = shift.pointer<float>();
		float scalef = scalep[j];
		float shiftf = shiftp[j];

		for (int i = 0; i < 2; ++i) { //B
			for (int k = 0; k < 2; ++k) { //Y
				for (int l = 0; l < 3; ++l) { //X
					float data = output_ptr[i + 2 * j + 2 * 2 * l + 2 * 2 * 3 * k];
					data = (data - shiftf) / scalef;
					sum += data;
					var += data * data;
				}
			}
		}
		sum /= 2 * 3 * 2; 
		var /= 2 * 3 * 2;

		EXPECT_NEAR(sum, 0, 1e-03F);
		EXPECT_NEAR(var, 1, 1e-03F);
	}
}

TEST(batch_normalization_gpu, basic_in2x3x2x2_with_var_mean_calc) {
    //  Mean   : 3x2x2
    //  Input  : 2x3x2x2
    //  Output : 2x3x2x2

    //  Input:
    //  f0: b0:  1    2  -10   b1:   0    0     -11
    //  f0: b0:  3    4  -14   b1:   0.5 -0.5   -15  
    //  f1: b0:  5    6  -12   b1:   1.5  5.2   -13     
    //  f1: b0:  7    8  -16   b1:   12   9     -17
    //
    //  Mean
    //  f0: -3.3333
    //  f1: -0.3583
    //
    //  Variance
    //  f0: 44.9305
    //  f1: 107.0624

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb,{ 2, 2, 3, 2 } });
    auto inv_variance = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 2, 1, 1 } });

    float epsilon = 0.0001f;

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(mutable_data("inv_variance", inv_variance));
    topology.add(batch_norm("batch_norm", "input", epsilon, "inv_variance"));

    set_values(input, {
        1.f, 0.f, 5.f, 1.5f,
        2.f, 0.f, 6.f, 5.2f,
        -10.f, -11.f, -12.f, -13.f,
        3.f, 0.5f, 7.f, 12.f,
        4.f, -0.5f, 8.f, 9.f,
        -14.f, -15.f, -16.f, -17.f
    });

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("batch_norm").get_memory();
    auto output_ptr = output.pointer<float>();

    for (int j = 0; j < 2; ++j) { //F
        float sum = 0, var = 0;
        for (int i = 0; i < 2; ++i) { //B
            for (int k = 0; k < 2; ++k) { //Y
                for (int l = 0; l < 3; ++l) { //X
                    float data = output_ptr[i + 2 * j + 2 * 2 * l + 2 * 2 * 3 * k];
                    sum += data;
                    var += data * data;
                }
            }
        }
        sum /= 2 * 3 * 2;
        var /= 2 * 3 * 2;

        EXPECT_NEAR(sum, 0, 1e-03F);
        EXPECT_NEAR(var, 1, 1e-03F);
    }
}

TEST(batch_normalization_gpu, basic_in2x3x2x2_with_var_mean_calc_no_inv_var) {
    //  Mean   : 3x2x2
    //  Input  : 2x3x2x2
    //  Output : 2x3x2x2

    //  Input:
    //  f0: b0:  1    2  -10   b1:   0    0     -11
    //  f0: b0:  3    4  -14   b1:   0.5 -0.5   -15  
    //  f1: b0:  5    6  -12   b1:   1.5  5.2   -13     
    //  f1: b0:  7    8  -16   b1:   12   9     -17
    //
    //  Mean
    //  f0: -3.3333
    //  f1: -0.3583
    //
    //  Variance
    //  f0: 44.9305
    //  f1: 107.0624

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb,{ 2, 2, 3, 2 } });

    float epsilon = 0.0001f;

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(batch_norm("batch_norm", "input", epsilon));

    set_values(input, {
        1.f, 0.f, 5.f, 1.5f,
        2.f, 0.f, 6.f, 5.2f,
        -10.f, -11.f, -12.f, -13.f,
        3.f, 0.5f, 7.f, 12.f,
        4.f, -0.5f, 8.f, 9.f,
        -14.f, -15.f, -16.f, -17.f
    });

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("batch_norm").get_memory();
    auto output_ptr = output.pointer<float>();

    for (int j = 0; j < 2; ++j) { //F
        float sum = 0, var = 0;
        for (int i = 0; i < 2; ++i) { //B
            for (int k = 0; k < 2; ++k) { //Y
                for (int l = 0; l < 3; ++l) { //X
                    float data = output_ptr[i + 2 * j + 2 * 2 * l + 2 * 2 * 3 * k];
                    sum += data;
                    var += data * data;
                }
            }
        }
        sum /= 2 * 3 * 2;
        var /= 2 * 3 * 2;

        EXPECT_NEAR(sum, 0, 1e-03F);
        EXPECT_NEAR(var, 1, 1e-03F);
    }
}

TEST(batch_normalization_gpu, basic_in2x3x2x2_with_var_mean_calc_scale_shift) {
	//  Mean   : 3x2x2
	//  Input  : 2x3x2x2
	//  Output : 2x3x2x2

	//  Input:
	//  f0: b0:  1    2  -10   b1:   0    0     -11
	//  f0: b0:  3    4  -14   b1:   0.5 -0.5   -15  
	//  f1: b0:  5    6  -12   b1:   1.5  5.2   -13     
	//  f1: b0:  7    8  -16   b1:   12   9     -17
	//
	//  Mean
	//  f0: -3.3333
	//  f1: -0.3583
	//
	//  Variance
	//  f0: 44.9305
	//  f1: 107.0624
	//
	//  Scale
	//  f0: 2.0
	//  f1: 1.0
	//
	//  Shift
	//  f0: 0.0
	//  f1: 5.0

	const auto& engine = get_test_engine();

	auto input = memory::allocate(engine, { data_types::f32, format::yxfb,{ 2, 2, 3, 2 } });
	auto mean = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 2, 1, 1 } });
	auto scale = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 2, 1, 1 } });
	auto shift = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 2, 1, 1 } });
	auto inv_variance = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 2, 1, 1 } });

	float epsilon = 0.0001f;

	topology topology;
	topology.add(input_layout("input", input.get_layout()));
	topology.add(data("scale", scale));
	topology.add(data("shift", shift));
	topology.add(mutable_data("inv_variance", inv_variance));
	topology.add(batch_norm("batch_norm", "input", epsilon, "scale", "shift", "inv_variance"));

	set_values(input, {
		1.f, 0.f, 5.f, 1.5f,
		2.f, 0.f, 6.f, 5.2f,
		-10.f, -11.f, -12.f, -13.f,
		3.f, 0.5f, 7.f, 12.f,
		4.f, -0.5f, 8.f, 9.f,
		-14.f, -15.f, -16.f, -17.f
	});

	set_values(scale, { 2.f, 1.f });
	set_values(shift, { 0.f, 5.f });

	network network(engine, topology);

	network.set_input_data("input", input);

	auto outputs = network.execute();

	auto output = outputs.at("batch_norm").get_memory();
	auto output_ptr = output.pointer<float>();

	for (int j = 0; j < 2; ++j) { //F
		float sum = 0, var = 0;

		auto scalep = scale.pointer<float>();
		auto shiftp = shift.pointer<float>();
		float scalef = scalep[j];
		float shiftf = shiftp[j];

		for (int i = 0; i < 2; ++i) { //B
			for (int k = 0; k < 2; ++k) { //Y
				for (int l = 0; l < 3; ++l) { //X
					float data = output_ptr[i + 2 * j + 2 * 2 * l + 2 * 2 * 3 * k];
					data = (data - shiftf) / scalef;
					sum += data;
					var += data * data;
				}
			}
		}
		sum /= 2 * 3 * 2;
		var /= 2 * 3 * 2;

		EXPECT_NEAR(sum, 0, 1e-03F);
		EXPECT_NEAR(var, 1, 1e-03F);
	}
}

TEST(batch_normalization_gpu, basic_in2x3x2x2_with_var_mean_calc_scale_shift_no_inv_var) {
    //  Mean   : 3x2x2
    //  Input  : 2x3x2x2
    //  Output : 2x3x2x2

    //  Input:
    //  f0: b0:  1    2  -10   b1:   0    0     -11
    //  f0: b0:  3    4  -14   b1:   0.5 -0.5   -15  
    //  f1: b0:  5    6  -12   b1:   1.5  5.2   -13     
    //  f1: b0:  7    8  -16   b1:   12   9     -17
    //
    //  Mean
    //  f0: -3.3333
    //  f1: -0.3583
    //
    //  Variance
    //  f0: 44.9305
    //  f1: 107.0624
    //
    //  Scale
    //  f0: 2.0
    //  f1: 1.0
    //
    //  Shift
    //  f0: 0.0
    //  f1: 5.0

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb,{ 2, 2, 3, 2 } });
    auto scale = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 2, 1, 1 } });
    auto shift = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 2, 1, 1 } });

    float epsilon = 0.0001f;

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(data("scale", scale));
    topology.add(data("shift", shift));
    topology.add(batch_norm("batch_norm", "input", epsilon, "scale", "shift"));

    set_values(input, {
        1.f, 0.f, 5.f, 1.5f,
        2.f, 0.f, 6.f, 5.2f,
        -10.f, -11.f, -12.f, -13.f,
        3.f, 0.5f, 7.f, 12.f,
        4.f, -0.5f, 8.f, 9.f,
        -14.f, -15.f, -16.f, -17.f
    });

    set_values(scale, { 2.f, 1.f });
    set_values(shift, { 0.f, 5.f });

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("batch_norm").get_memory();
    auto output_ptr = output.pointer<float>();

    for (int j = 0; j < 2; ++j) { //F
        float sum = 0, var = 0;

        auto scalep = scale.pointer<float>();
        auto shiftp = shift.pointer<float>();
        float scalef = scalep[j];
        float shiftf = shiftp[j];

        for (int i = 0; i < 2; ++i) { //B
            for (int k = 0; k < 2; ++k) { //Y
                for (int l = 0; l < 3; ++l) { //X
                    float data = output_ptr[i + 2 * j + 2 * 2 * l + 2 * 2 * 3 * k];
                    data = (data - shiftf) / scalef;
                    sum += data;
                    var += data * data;
                }
            }
        }
        sum /= 2 * 3 * 2;
        var /= 2 * 3 * 2;

        EXPECT_NEAR(sum, 0, 1e-03F);
        EXPECT_NEAR(var, 1, 1e-03F);
    }
}

TEST(batch_normalization_gpu, basic_in2x3x2x2_with_var_mean_outputs) {
	//  Mean   : 3x2x2
	//  Input  : 2x3x2x2
	//  Output : 2x3x2x2

	//  Input:
	//  f0: b0:  1    2  -10   b1:   0    0     -11
	//  f0: b0:  3    4  -14   b1:   0.5 -0.5   -15  
	//  f1: b0:  5    6  -12   b1:   1.5  5.2   -13     
	//  f1: b0:  7    8  -16   b1:   12   9     -17
	//
	//  Mean (to be calculated)
	//  f0: -3.3333
	//  f1: -0.3583
	//
	//  Variance (to be calculated)
	//  f0: 44.9305
	//  f1: 107.0624
	//
	//  Scale
	//  f0: 2.0
	//  f1: 1.0
	//
	//  Shift
	//  f0: 0.0
	//  f1: 5.0

	const auto& engine = get_test_engine();

	auto input = memory::allocate(engine, { data_types::f32, format::yxfb,{ 2, 2, 3, 2 } });
	auto mean_out = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 2, 1, 1 } });
	auto variance_out = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 2, 1, 1 } });
	auto scale = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 2, 1, 1 } });
	auto shift = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 2, 1, 1 } });
	auto inv_variance = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 2, 1, 1 } });

	float epsilon = 0.0001f;

	topology topology;
	topology.add(input_layout("input", input.get_layout()));
	topology.add(data("scale", scale));
	topology.add(data("shift", shift));
	topology.add(mutable_data("mean_out", mean_out));
	topology.add(mutable_data("variance_out", variance_out));
	topology.add(mutable_data("inv_variance", inv_variance));
	topology.add(batch_norm("batch_norm", "input", epsilon, "mean_out", "variance_out", "scale", "shift", "inv_variance"));

	set_values(input, {
		1.f, 0.f, 5.f, 1.5f,
		2.f, 0.f, 6.f, 5.2f,
		-10.f, -11.f, -12.f, -13.f,
		3.f, 0.5f, 7.f, 12.f,
		4.f, -0.5f, 8.f, 9.f,
		-14.f, -15.f, -16.f, -17.f
	});

	set_values(scale, { 2.f, 1.f });
	set_values(shift, { 0.f, 5.f });

	network network(engine, topology);

	network.set_input_data("input", input);

	auto outputs = network.execute();

	auto output = outputs.at("batch_norm").get_memory();
	auto output_ptr = output.pointer<float>();

	std::vector<float> mean_ref = { -3.3333f, -0.3583f };
	std::vector<float> val_ref = { 44.9305f, 107.0624f };

	for (int j = 0; j < 2; ++j) { //F
		float sum = 0, var = 0;

		auto scalep = scale.pointer<float>();
		auto shiftp = shift.pointer<float>();
		float scalef = scalep[j];
		float shiftf = shiftp[j];

		auto meanp = mean_out.pointer<float>();
		auto varp = variance_out.pointer<float>();
		float meanf = meanp[j];
		float varf = varp[j];

		for (int i = 0; i < 2; ++i) { //B
			for (int k = 0; k < 2; ++k) { //Y
				for (int l = 0; l < 3; ++l) { //X
					float data = output_ptr[i + 2 * j + 2 * 2 * l + 2 * 2 * 3 * k];
					data = (data - shiftf) / scalef;
					sum += data;
					var += data * data;
				}
			}
		}
		sum /= 2 * 3 * 2;
		var /= 2 * 3 * 2;

		EXPECT_NEAR(sum, 0, 1e-03F);
		EXPECT_NEAR(var, 1, 1e-03F);

		EXPECT_NEAR(meanf, mean_ref[j], 1e-03F);
		EXPECT_NEAR(varf, val_ref[j], 1e-03F);
	}
}

TEST(batch_normalization_gpu, basic_in2x3x2x2_with_var_mean_outputs_no_inv_var) {
    //  Mean   : 3x2x2
    //  Input  : 2x3x2x2
    //  Output : 2x3x2x2

    //  Input:
    //  f0: b0:  1    2  -10   b1:   0    0     -11
    //  f0: b0:  3    4  -14   b1:   0.5 -0.5   -15  
    //  f1: b0:  5    6  -12   b1:   1.5  5.2   -13     
    //  f1: b0:  7    8  -16   b1:   12   9     -17
    //
    //  Mean (to be calculated)
    //  f0: -3.3333
    //  f1: -0.3583
    //
    //  Variance (to be calculated)
    //  f0: 44.9305
    //  f1: 107.0624
    //
    //  Scale
    //  f0: 2.0
    //  f1: 1.0
    //
    //  Shift
    //  f0: 0.0
    //  f1: 5.0

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb,{ 2, 2, 3, 2 } });
    auto mean_out = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 2, 1, 1 } });
    auto variance_out = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 2, 1, 1 } });
    auto scale = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 2, 1, 1 } });
    auto shift = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 2, 1, 1 } });

    float epsilon = 0.0001f;

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(data("scale", scale));
    topology.add(data("shift", shift));
    topology.add(mutable_data("mean_out", mean_out));
    topology.add(mutable_data("variance_out", variance_out));
    topology.add(batch_norm("batch_norm", "input", epsilon, "mean_out", "variance_out", "scale", "shift"));

    set_values(input, {
        1.f, 0.f, 5.f, 1.5f,
        2.f, 0.f, 6.f, 5.2f,
        -10.f, -11.f, -12.f, -13.f,
        3.f, 0.5f, 7.f, 12.f,
        4.f, -0.5f, 8.f, 9.f,
        -14.f, -15.f, -16.f, -17.f
    });

    set_values(scale, { 2.f, 1.f });
    set_values(shift, { 0.f, 5.f });

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("batch_norm").get_memory();
    auto output_ptr = output.pointer<float>();

    std::vector<float> mean_ref = { -3.3333f, -0.3583f };
    std::vector<float> val_ref = { 44.9305f, 107.0624f };

    for (int j = 0; j < 2; ++j) { //F
        float sum = 0, var = 0;

        auto scalep = scale.pointer<float>();
        auto shiftp = shift.pointer<float>();
        float scalef = scalep[j];
        float shiftf = shiftp[j];

        auto meanp = mean_out.pointer<float>();
        auto varp = variance_out.pointer<float>();
        float meanf = meanp[j];
        float varf = varp[j];

        for (int i = 0; i < 2; ++i) { //B
            for (int k = 0; k < 2; ++k) { //Y
                for (int l = 0; l < 3; ++l) { //X
                    float data = output_ptr[i + 2 * j + 2 * 2 * l + 2 * 2 * 3 * k];
                    data = (data - shiftf) / scalef;
                    sum += data;
                    var += data * data;
                }
            }
        }
        sum /= 2 * 3 * 2;
        var /= 2 * 3 * 2;

        EXPECT_NEAR(sum, 0, 1e-03F);
        EXPECT_NEAR(var, 1, 1e-03F);

        EXPECT_NEAR(meanf, mean_ref[j], 1e-03F);
        EXPECT_NEAR(varf, val_ref[j], 1e-03F);
    }
}

TEST(batch_normalization_gpu, basic_in2x3x2x2_with_var_mean_outputs_error_out_type) {
	const auto& engine = get_test_engine();

	auto input = memory::allocate(engine, { data_types::f32, format::yxfb,{ 2, 2, 3, 2 } });
	auto mean_out = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 2, 1, 1 } });
	auto variance_out = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 2, 1, 1 } });
	auto scale = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 2, 1, 1 } });
	auto shift = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 2, 1, 1 } });
	auto inv_variance = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 2, 1, 1 } });

	float epsilon = 0.0001f;

	topology topology;
	topology.add(input_layout("input", input.get_layout()));
	topology.add(data("scale", scale));
	topology.add(data("shift", shift));
	topology.add(data("mean_out", mean_out));
	topology.add(data("variance_out", variance_out));
	topology.add(data("inv_variance", inv_variance));
	topology.add(batch_norm("batch_norm", "input", epsilon, "mean_out", "variance_out", "scale", "shift", "inv_variance"));

	EXPECT_ANY_THROW(network(engine, topology));
}

TEST(batch_normalization_gpu, basic_in2x3x2x2_with_var_mean_outputs_error_non_equal_types) {
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb,{ 2, 2, 3, 2 } });
    auto mean_out = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 2, 1, 1 } });
    auto variance_out = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 2, 1, 1 } });
    auto scale = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 2, 1, 1 } });
    auto shift = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 2, 1, 1 } });
    auto inv_variance = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 2, 1, 1 } });

    float epsilon = 0.0001f;

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(data("scale", scale));
    topology.add(data("shift", shift));
    topology.add(data("mean_out", mean_out));
    topology.add(mutable_data("variance_out", variance_out));
    topology.add(mutable_data("inv_variance", inv_variance));
    topology.add(batch_norm("batch_norm", "input", epsilon, "mean_out", "variance_out", "scale", "shift", "inv_variance"));

    EXPECT_ANY_THROW(network(engine, topology));
}

TEST(batch_normalization_gpu, basic_in2x2x3x2_bfyx) {
    //  Mean   : 3x2x2
    //  Input  : 2x3x2x2
    //  Output : 2x3x2x2

    //  Input:
    //  f0: b0:  1    2  -10   b1:   0    0     -11
    //  f0: b0:  3    4  -14   b1:   0.5 -0.5   -15  
    //  f1: b0:  5    6  -12   b1:   1.5  5.2   -13     
    //  f1: b0:  7    8  -16   b1:   12   9     -17
    //
    //  Mean
    //  f0: -3.3333
    //  f1: -0.3583
    //
    //  Variance
    //  f0: 44.9305
    //  f1: 107.0624

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 2, 2, 3, 2 } });
    auto mean = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 2, 1, 1 } });
    auto variance = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 2, 1, 1 } });

    float epsilon = 0.0001f;

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(data("mean", mean));
    topology.add(data("variance", variance));
    topology.add(batch_norm("batch_norm", "input", "mean", "variance", epsilon));

    set_values(input, {
        1.f, 2.f, -10.f, 3.f,
        4.f, -14.f, 5.f, 6.f,
        -12.f, 7.f, 8.f, -16.f,
        0.f, 0.f, -11.f, 0.5f,
        -0.5f, -15.f, 1.5f, 5.2f,
        -13.f, 12.f, 9.f, -17.f
    });

    //  f0: b0:  1    2  -10   b1:   0    0     -11
    //  f0: b0:  3    4  -14   b1:   0.5 -0.5   -15  
    //  f1: b0:  5    6  -12   b1:   1.5  5.2   -13     
    //  f1: b0:  7    8  -16   b1:   12   9     -17

    set_values(mean, { -3.3333f, -0.3583f });
    set_values(variance, { 44.9305f, 107.0624f });

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("batch_norm").get_memory();
    auto output_ptr = output.pointer<float>();

    for (int j = 0; j < 2; ++j) { //F
        float sum = 0, var = 0;
        for (int i = 0; i < 2; ++i) { //B
            for (int k = 0; k < 2; ++k) { //Y
                for (int l = 0; l < 3; ++l) { //X
                    float data = output_ptr[l + k * 3 + j * 2 * 3 + i * 2 * 2 * 3];
                    sum += data;
                    var += data * data;
                }
            }
        }
        sum /= 2 * 3 * 2;
        var /= 2 * 3 * 2;

        EXPECT_NEAR(sum, 0, 1e-03F);
        EXPECT_NEAR(var, 1, 1e-03F);
    }
}

TEST(batch_normalization_gpu, basic_in2x2x3x2_bfyx_padding) {
    //  Mean   : 3x2x2
    //  Input  : 2x3x2x2
    //  Output : 2x3x2x2
    //  Input padding : 1x2
    //  Output padding : 2x1

    //  Input:
    //  f0: b0:  1    2  -10   b1:   0    0     -11
    //  f0: b0:  3    4  -14   b1:   0.5 -0.5   -15  
    //  f1: b0:  5    6  -12   b1:   1.5  5.2   -13     
    //  f1: b0:  7    8  -16   b1:   12   9     -17
    //
    //  Mean
    //  f0: -3.3333
    //  f1: -0.3583
    //
    //  Variance
    //  f0: 44.9305
    //  f1: 107.0624

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 2, 2, 3, 2 } });
    auto mean = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 2, 1, 1 } });
    auto variance = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 2, 1, 1 } });

    float epsilon = 0.0001f;

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(data("mean", mean));
    topology.add(data("variance", variance));
    topology.add(reorder("reorder", "input", input.get_layout().with_padding(padding{ { 0, 0, 1, 2 }, 0 })));
    topology.add(batch_norm("batch_norm", "reorder", "mean", "variance", epsilon, padding({ 0, 0, 2, 1 }, 0)));

    set_values(input, {
        1.f, 2.f, -10.f, 3.f,
        4.f, -14.f, 5.f, 6.f,
        -12.f, 7.f, 8.f, -16.f,
        0.f, 0.f, -11.f, 0.5f,
        -0.5f, -15.f, 1.5f, 5.2f,
        -13.f, 12.f, 9.f, -17.f
    });

    //  f0: b0:  1    2  -10   b1:   0    0     -11
    //  f0: b0:  3    4  -14   b1:   0.5 -0.5   -15  
    //  f1: b0:  5    6  -12   b1:   1.5  5.2   -13     
    //  f1: b0:  7    8  -16   b1:   12   9     -17

    set_values(mean, { -3.3333f, -0.3583f });
    set_values(variance, { 44.9305f, 107.0624f });

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("batch_norm").get_memory();
    auto output_ptr = output.pointer<float>();

    for (int j = 0; j < 2; ++j) { //F
        float sum = 0, var = 0;
        for (int i = 0; i < 2; ++i) { //B
            for (int k = 0; k < 2; ++k) { //Y
                for (int l = 0; l < 3; ++l) { //X
                    float data = output_ptr[l + 2 + 7 * (k + 1 + 4 * (j + 2 * i))];
                    sum += data;
                    var += data * data;
                }
            }
        }
        sum /= 2 * 3 * 2;
        var /= 2 * 3 * 2;

        EXPECT_NEAR(sum, 0, 1e-03F);
        EXPECT_NEAR(var, 1, 1e-03F);
    }
}

TEST(batch_normalization_gpu, basic_to_string) {
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb,{ 2, 2, 3, 2 } });

    auto mean = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 2, 1, 1 } });
    auto variance = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 2, 1, 1 } });

    auto scale = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 2, 1, 1 } });
    auto shift = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 2, 1, 1 } });

    auto inv_variance = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 2, 1, 1 } });

    auto mean_out = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 2, 1, 1 } });
    auto variance_out = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 2, 1, 1 } });

    float epsilon = 0.0001f;

    topology topology;
    topology.add(input_layout("input", input.get_layout()));

    topology.add(data("mean", mean));
    topology.add(data("variance", variance));

    topology.add(data("scale", scale));
    topology.add(data("shift", shift));

    topology.add(mutable_data("inv_variance", inv_variance));

    topology.add(mutable_data("mean_out", mean_out));
    topology.add(mutable_data("variance_out", variance_out));

    topology.add(batch_norm("batch_norm0", "input", "mean", "variance", epsilon));
    topology.add(batch_norm("batch_norm1", "input", "mean", "variance", "scale", "shift", epsilon));
    topology.add(batch_norm("batch_norm2", "input", epsilon));
    topology.add(batch_norm("batch_norm3", "input", epsilon, "inv_variance"));
    topology.add(batch_norm("batch_norm4", "input", epsilon, "scale", "shift"));
    topology.add(batch_norm("batch_norm5", "input", epsilon, "scale", "shift", "inv_variance"));
    topology.add(batch_norm("batch_norm6", "input", epsilon, "mean_out", "variance_out", "scale", "shift" ));
    topology.add(batch_norm("batch_norm7", "input", epsilon, "mean_out", "variance_out", "scale", "shift", "inv_variance"));

    network network(engine, topology);

    size_t zero_length = 0;

    EXPECT_NE(network.get_primitive_info("batch_norm0").length(), zero_length);
    EXPECT_NE(network.get_primitive_info("batch_norm1").length(), zero_length);
    EXPECT_NE(network.get_primitive_info("batch_norm2").length(), zero_length);
    EXPECT_NE(network.get_primitive_info("batch_norm3").length(), zero_length);
    EXPECT_NE(network.get_primitive_info("batch_norm4").length(), zero_length);
    EXPECT_NE(network.get_primitive_info("batch_norm5").length(), zero_length);
    EXPECT_NE(network.get_primitive_info("batch_norm6").length(), zero_length);
    EXPECT_NE(network.get_primitive_info("batch_norm7").length(), zero_length);
}                                         

TEST(batch_normalization_gpu, basic_in2x3x2x2_yxfb_scale_shift_different_shapes) {
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb,{ 2, 2, 3, 2 } });
    auto mean = memory::allocate(engine, { data_types::f32, format::yxfb,{ 2, 1, 1, 1 } });
    auto variance = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 2, 1, 1 } });
    auto scale = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 1, 2, 1 } });
    auto shift = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 1, 1, 2 } });

    float epsilon = 0.0001f;

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(data("mean", mean));
    topology.add(data("variance", variance));
    topology.add(data("scale", scale));
    topology.add(data("shift", shift));
    topology.add(batch_norm("batch_norm", "input", "mean", "variance", "scale", "shift", epsilon));

    set_values(input, {
        1.f, 0.f, 5.f, 1.5f,
        2.f, 0.f, 6.f, 5.2f,
        -10.f, -11.f, -12.f, -13.f,
        3.f, 0.5f, 7.f, 12.f,
        4.f, -0.5f, 8.f, 9.f,
        -14.f, -15.f, -16.f, -17.f
    });

    set_values(mean, { -3.3333f, -0.3583f });
    set_values(variance, { 44.9305f, 107.0624f });
    set_values(scale, { 2.f, 1.f });
    set_values(shift, { 0.f, 5.f });

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("batch_norm").get_memory();
    auto output_ptr = output.pointer<float>();

    for (int j = 0; j < 2; ++j) { //F
        float sum = 0, var = 0;

        auto scalep = scale.pointer<float>();
        auto shiftp = shift.pointer<float>();
        float scalef = scalep[j];
        float shiftf = shiftp[j];

        for (int i = 0; i < 2; ++i) { //B
            for (int k = 0; k < 2; ++k) { //Y
                for (int l = 0; l < 3; ++l) { //X
                    float data = output_ptr[i + 2 * j + 2 * 2 * l + 2 * 2 * 3 * k];
                    data = (data - shiftf) / scalef;
                    sum += data;
                    var += data * data;
                }
            }
        }
        sum /= 2 * 3 * 2;
        var /= 2 * 3 * 2;

        EXPECT_NEAR(sum, 0, 1e-03F);
        EXPECT_NEAR(var, 1, 1e-03F);
    }
}

TEST(batch_normalization_gpu, basic_in2x3x2x2_yxfb_scale_shift_different_shapes_input_layouts) {
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb,{ 2, 2, 3, 2 } });
    auto mean = memory::allocate(engine, { data_types::f32, format::yxfb,{ 2, 1, 1, 1 } });
    auto variance = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 2, 1, 1 } });
    auto scale = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 1, 2, 1 } });
    auto shift = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 1, 1, 2 } });

    float epsilon = 0.0001f;

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("mean", mean.get_layout()));
    topology.add(input_layout("variance", variance.get_layout()));
    topology.add(input_layout("scale", scale.get_layout()));
    topology.add(input_layout("shift", shift.get_layout()));
    topology.add(batch_norm("batch_norm", "input", "mean", "variance", "scale", "shift", epsilon));

    set_values(input, {
        1.f, 0.f, 5.f, 1.5f,
        2.f, 0.f, 6.f, 5.2f,
        -10.f, -11.f, -12.f, -13.f,
        3.f, 0.5f, 7.f, 12.f,
        4.f, -0.5f, 8.f, 9.f,
        -14.f, -15.f, -16.f, -17.f
    });

    set_values(mean, { -3.3333f, -0.3583f });
    set_values(variance, { 44.9305f, 107.0624f });
    set_values(scale, { 2.f, 1.f });
    set_values(shift, { 0.f, 5.f });

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("mean", mean);
    network.set_input_data("variance", variance);
    network.set_input_data("scale", scale);
    network.set_input_data("shift", shift);

    auto outputs = network.execute();

    auto output = outputs.at("batch_norm").get_memory();
    auto output_ptr = output.pointer<float>();

    for (int j = 0; j < 2; ++j) { //F
        float sum = 0, var = 0;

        auto scalep = scale.pointer<float>();
        auto shiftp = shift.pointer<float>();
        float scalef = scalep[j];
        float shiftf = shiftp[j];

        for (int i = 0; i < 2; ++i) { //B
            for (int k = 0; k < 2; ++k) { //Y
                for (int l = 0; l < 3; ++l) { //X
                    float data = output_ptr[i + 2 * j + 2 * 2 * l + 2 * 2 * 3 * k];
                    data = (data - shiftf) / scalef;
                    sum += data;
                    var += data * data;
                }
            }
        }
        sum /= 2 * 3 * 2;
        var /= 2 * 3 * 2;

        EXPECT_NEAR(sum, 0, 1e-03F);
        EXPECT_NEAR(var, 1, 1e-03F);
    }
}

TEST(batch_normalization_gpu, basic_in2x3x2x2_yxfb_with_var_mean_outputs_no_inv_var_different_shapes) {
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb,{ 2, 2, 3, 2 } });
    auto mean_out = memory::allocate(engine, { data_types::f32, format::yxfb,{ 2, 1, 1, 1 } });
    auto variance_out = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 2, 1, 1 } });
    auto scale = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 1, 2, 1 } });
    auto shift = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 1, 1, 2 } });

    float epsilon = 0.0001f;

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(data("scale", scale));
    topology.add(data("shift", shift));
    topology.add(mutable_data("mean_out", mean_out));
    topology.add(mutable_data("variance_out", variance_out));
    topology.add(batch_norm("batch_norm", "input", epsilon, "mean_out", "variance_out", "scale", "shift"));

    set_values(input, {
        1.f, 0.f, 5.f, 1.5f,
        2.f, 0.f, 6.f, 5.2f,
        -10.f, -11.f, -12.f, -13.f,
        3.f, 0.5f, 7.f, 12.f,
        4.f, -0.5f, 8.f, 9.f,
        -14.f, -15.f, -16.f, -17.f
    });

    set_values(scale, { 2.f, 1.f });
    set_values(shift, { 0.f, 5.f });

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("batch_norm").get_memory();
    auto output_ptr = output.pointer<float>();

    std::vector<float> mean_ref = { -3.3333f, -0.3583f };
    std::vector<float> val_ref = { 44.9305f, 107.0624f };

    for (int j = 0; j < 2; ++j) { //F
        float sum = 0, var = 0;

        auto scalep = scale.pointer<float>();
        auto shiftp = shift.pointer<float>();
        float scalef = scalep[j];
        float shiftf = shiftp[j];

        auto meanp = mean_out.pointer<float>();
        auto varp = variance_out.pointer<float>();
        float meanf = meanp[j];
        float varf = varp[j];

        for (int i = 0; i < 2; ++i) { //B
            for (int k = 0; k < 2; ++k) { //Y
                for (int l = 0; l < 3; ++l) { //X
                    float data = output_ptr[i + 2 * j + 2 * 2 * l + 2 * 2 * 3 * k];
                    data = (data - shiftf) / scalef;
                    sum += data;
                    var += data * data;
                }
            }
        }
        sum /= 2 * 3 * 2;
        var /= 2 * 3 * 2;

        EXPECT_NEAR(sum, 0, 1e-03F);
        EXPECT_NEAR(var, 1, 1e-03F);

        EXPECT_NEAR(meanf, mean_ref[j], 1e-03F);
        EXPECT_NEAR(varf, val_ref[j], 1e-03F);
    }
}

TEST(batch_normalization_gpu, basic_in2x2x3x2_byxf_scale_shift_different_shapes) {
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::byxf,{ 2, 2, 3, 2 } });
    auto mean = memory::allocate(engine, { data_types::f32, format::byxf,{ 2, 1, 1, 1 } });
    auto variance = memory::allocate(engine, { data_types::f32, format::byxf,{ 1, 2, 1, 1 } });
    auto scale = memory::allocate(engine, { data_types::f32, format::byxf,{ 1, 1, 2, 1 } });
    auto shift = memory::allocate(engine, { data_types::f32, format::byxf,{ 1, 1, 1, 2 } });

    float epsilon = 0.0001f;

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(data("mean", mean));
    topology.add(data("variance", variance));
    topology.add(data("scale", scale));
    topology.add(data("shift", shift));
    topology.add(batch_norm("batch_norm", "input", "mean", "variance", "scale", "shift", epsilon));

    set_values(input, {
        1.f, 5.f, 2.f, 6.f, -10.f, -12.f, 
        3.f, 7.f, 4.f, 8.f, -14.f, -16.f, 
        0.f, 1.5f, 0.f, 5.2f, -11.f, -13.f, 
        0.5f, 12.f, -0.5f, 9.f, -15.f, -17.f
    });

    set_values(mean, { -3.3333f, -0.3583f });
    set_values(variance, { 44.9305f, 107.0624f });
    set_values(scale, { 2.f, 1.f });
    set_values(shift, { 0.f, 5.f });

    std::vector<float> expected_result{
        0.646469f, 0.517855f, 0.795655f, 0.614501f, -0.99458f, -1.12512f, 
        0.944842f, 0.711146f, 1.09403f, 0.807792f, -1.59133f, -1.5117f, 
        0.497283f, 0.179596f, 0.497283f, 0.537184f, -1.14377f, -1.22176f, 
        0.571876f, 1.19437f, 0.42269f, 0.904437f, -1.74051f, -1.60834f
    };

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("batch_norm").get_memory();
    auto output_ptr = output.pointer<float>();

    for (int j = 0; j < 2; ++j) { //F
        float sum = 0, var = 0;

        auto scalep = scale.pointer<float>();
        auto shiftp = shift.pointer<float>();
        float scalef = scalep[j];
        float shiftf = shiftp[j];

        for (int i = 0; i < 2; ++i) { //B
            for (int k = 0; k < 2; ++k) { //Y
                for (int l = 0; l < 3; ++l) { //X
                    auto index = 12 * i + 6 * k + 2 * l + j;
                    float data = output_ptr[index];
                    data = (data - shiftf) / scalef;
                    EXPECT_NEAR(data, expected_result[index], 1e-3F);
                    sum += data;
                    var += data * data;
                }
            }
        }
        sum /= 2 * 3 * 2;
        var /= 2 * 3 * 2;

        EXPECT_NEAR(sum, 0, 1e-03F);
        EXPECT_NEAR(var, 1, 1e-03F);
    }
}

TEST(batch_normalization_gpu, basic_in2x2x3x2_byxf_with_var_mean_outputs_no_inv_var_different_shapes) {
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::byxf,{ 2, 2, 3, 2 } });
    auto mean_out = memory::allocate(engine, { data_types::f32, format::byxf,{ 2, 1, 1, 1 } });
    auto variance_out = memory::allocate(engine, { data_types::f32, format::byxf,{ 1, 2, 1, 1 } });
    auto scale = memory::allocate(engine, { data_types::f32, format::byxf,{ 1, 1, 2, 1 } });
    auto shift = memory::allocate(engine, { data_types::f32, format::byxf,{ 1, 1, 1, 2 } });

    float epsilon = 0.0001f;

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(data("scale", scale));
    topology.add(data("shift", shift));
    topology.add(mutable_data("mean_out", mean_out));
    topology.add(mutable_data("variance_out", variance_out));
    topology.add(batch_norm("batch_norm", "input", epsilon, "mean_out", "variance_out", "scale", "shift"));

    set_values(input, {
        1.f, 5.f, 2.f, 6.f, -10.f, -12.f,
        3.f, 7.f, 4.f, 8.f, -14.f, -16.f,
        0.f, 1.5f, 0.f, 5.2f, -11.f, -13.f,
        0.5f, 12.f, -0.5f, 9.f, -15.f, -17.f
    });

    set_values(scale, { 2.f, 1.f });
    set_values(shift, { 0.f, 5.f });

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("batch_norm").get_memory();
    auto output_ptr = output.pointer<float>();

    std::vector<float> mean_ref = { -3.3333f, -0.3583f };
    std::vector<float> val_ref = { 44.9305f, 107.0624f };

    std::vector<float> expected_result{
        0.646469f, 0.517855f, 0.795655f, 0.614501f, -0.99458f, -1.12512f,
        0.944842f, 0.711146f, 1.09403f, 0.807792f, -1.59133f, -1.5117f,
        0.497283f, 0.179596f, 0.497283f, 0.537184f, -1.14377f, -1.22176f,
        0.571876f, 1.19437f, 0.42269f, 0.904437f, -1.74051f, -1.60834f
    };

    for (int j = 0; j < 2; ++j) { //F
        float sum = 0, var = 0;

        auto scalep = scale.pointer<float>();
        auto shiftp = shift.pointer<float>();
        float scalef = scalep[j];
        float shiftf = shiftp[j];

        auto meanp = mean_out.pointer<float>();
        auto varp = variance_out.pointer<float>();
        float meanf = meanp[j];
        float varf = varp[j];

        for (int i = 0; i < 2; ++i) { //B
            for (int k = 0; k < 2; ++k) { //Y
                for (int l = 0; l < 3; ++l) { //X
                    auto index = 12 * i + 6 * k + 2 * l + j;
                    float data = output_ptr[index];
                    data = (data - shiftf) / scalef;
                    EXPECT_NEAR(data, expected_result[index], 1e-3F);
                    sum += data;
                    var += data * data;
                }
            }
        }
        sum /= 2 * 3 * 2;
        var /= 2 * 3 * 2;

        EXPECT_NEAR(sum, 0, 1e-03F);
        EXPECT_NEAR(var, 1, 1e-03F);

        EXPECT_NEAR(meanf, mean_ref[j], 1e-03F);
        EXPECT_NEAR(varf, val_ref[j], 1e-03F);
    }
}

TEST(batch_normalization_gpu, basic_in2x3x5x2_yxfb_scale_shift_different_shapes) {
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb,{ 2, 5, 3, 2 } });
    auto mean = memory::allocate(engine, { data_types::f32, format::yxfb,{ 5, 1, 1, 1 } });
    auto variance = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 5, 1, 1 } });
    auto scale = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 1, 5, 1 } });
    auto shift = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 1, 1, 5 } });

    float epsilon = 0.0001f;

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(data("mean", mean));
    topology.add(data("variance", variance));
    topology.add(data("scale", scale));
    topology.add(data("shift", shift));
    topology.add(batch_norm("batch_norm", "input", "mean", "variance", "scale", "shift", epsilon));

    set_values(input, {
        // y0x0
        1.f, 0.f, // f0
        5.f, 1.5f, // f1
        1.f, 0.f, // f2
        5.f, 1.5f, // f3
        1.f, 0.f, // f4

        // y0x1
        2.f, 0.f, 
        6.f, 5.2f,
        2.f, 0.f,
        6.f, 5.2f,
        2.f, 0.f,

        // y0x2
        -10.f, -11.f, 
        -12.f, -13.f,
        -10.f, -11.f,
        -12.f, -13.f,
        -10.f, -11.f,

        // y1x0
        3.f, 0.5f, 
        7.f, 12.f,
        3.f, 0.5f,
        7.f, 12.f,
        3.f, 0.5f,

        // y1x1
        4.f, -0.5f, 
        8.f, 9.f,
        4.f, -0.5f,
        8.f, 9.f,
        4.f, -0.5f,

        // y1x2
        -14.f, -15.f,
        -16.f, -17.f,
        -14.f, -15.f,
        -16.f, -17.f,
        - 14.f, -15.f
    });

    set_values(mean, { -3.3333f, -0.3583f, -3.3333f, -0.3583f, -3.3333f });
    set_values(variance, { 44.9305f, 107.0624f, 44.9305f, 107.0624f, 44.9305f });
    set_values(scale, { 2.f, 1.f, 3.f, 4.f, 5.f });
    set_values(shift, { 0.f, 5.f, -5.f, -15.f, 0.5f });

    std::vector<float> expected_result{
        0.646469f, 0.497283f, 
        0.517855f, 0.179596f, 
        0.646469f, 0.497283f, 
        0.517855f, 0.179596f, 
        0.646469f, 0.497283f, 
        
        0.795655f, 0.497283f, 
        0.614501f, 0.537184f, 
        0.795655f, 0.497283f, 
        0.614501f, 0.537184f, 
        0.795655f, 0.497283f, 
        
        -0.99458f, -1.14377f, 
        -1.12512f, -1.22176f, 
        -0.99458f, -1.14377f, 
        -1.12512f, -1.22176f, 
        -0.99458f, -1.14377f, 
        
        0.944842f, 0.571876f, 
        0.711146f, 1.19437f, 
        0.944842f, 0.571876f, 
        0.711146f, 1.19437f, 
        0.944842f, 0.571876f, 
        
        1.09403f, 0.42269f, 
        0.807792f, 0.904437f, 
        1.09403f, 0.42269f, 
        0.807792f, 0.904437f, 
        1.09403f, 0.42269f, 
        
        -1.59133f, -1.74051f, 
        -1.5117f, -1.60834f, 
        -1.59133f, -1.74051f, 
        -1.5117f, -1.60834f, 
        -1.59133f, -1.74051f
    };

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("batch_norm").get_memory();
    auto output_ptr = output.pointer<float>();

    for (int j = 0; j < 5; ++j) { //F
        float sum = 0, var = 0;

        auto scalep = scale.pointer<float>();
        auto shiftp = shift.pointer<float>();
        float scalef = scalep[j];
        float shiftf = shiftp[j];

        for (int i = 0; i < 2; ++i) { //B
            for (int k = 0; k < 2; ++k) { //Y
                for (int l = 0; l < 3; ++l) { //X
                    int index = 30 * k + 10 * l + 2 * j + i;
                    float data = output_ptr[index];
                    data = (data - shiftf) / scalef;
                    EXPECT_NEAR(data, expected_result[index], 1e-3F);
                    sum += data;
                    var += data * data;
                }
            }
        }
        sum /= 2 * 3 * 2;
        var /= 2 * 3 * 2;

        EXPECT_NEAR(sum, 0, 1e-03F);
        EXPECT_NEAR(var, 1, 1e-03F);
    }
}

TEST(batch_normalization_gpu, basic_in2x3x5x2_yxfb_with_var_mean_outputs_no_inv_var_different_shapes) {
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb,{ 2, 5, 3, 2 } });
    auto mean_out = memory::allocate(engine, { data_types::f32, format::yxfb,{ 5, 1, 1, 1 } });
    auto variance_out = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 5, 1, 1 } });
    auto scale = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 1, 5, 1 } });
    auto shift = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 1, 1, 5 } });

    float epsilon = 0.0001f;

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(data("scale", scale));
    topology.add(data("shift", shift));
    topology.add(mutable_data("mean_out", mean_out));
    topology.add(mutable_data("variance_out", variance_out));
    topology.add(batch_norm("batch_norm", "input", epsilon, "mean_out", "variance_out", "scale", "shift"));

    set_values(input, {
        // y0x0
        1.f, 0.f, // f0
        5.f, 1.5f, // f1
        1.f, 0.f, // f2
        5.f, 1.5f, // f3
        1.f, 0.f, // f4

        // y0x1
        2.f, 0.f,
        6.f, 5.2f,
        2.f, 0.f,
        6.f, 5.2f,
        2.f, 0.f,

        // y0x2
        -10.f, -11.f,
        -12.f, -13.f,
        -10.f, -11.f,
        -12.f, -13.f,
        -10.f, -11.f,

        // y1x0
        3.f, 0.5f,
        7.f, 12.f,
        3.f, 0.5f,
        7.f, 12.f,
        3.f, 0.5f,

        // y1x1
        4.f, -0.5f,
        8.f, 9.f,
        4.f, -0.5f,
        8.f, 9.f,
        4.f, -0.5f,

        // y1x2
        -14.f, -15.f,
        -16.f, -17.f,
        -14.f, -15.f,
        -16.f, -17.f,
        -14.f, -15.f
    });

    set_values(scale, { 2.f, 1.f, 3.f, 4.f, 5.f });
    set_values(shift, { 0.f, 5.f, -5.f, -15.f, 0.5f });

    std::vector<float> expected_result{
        0.646469f, 0.497283f,
        0.517855f, 0.179596f,
        0.646469f, 0.497283f,
        0.517855f, 0.179596f,
        0.646469f, 0.497283f,

        0.795655f, 0.497283f,
        0.614501f, 0.537184f,
        0.795655f, 0.497283f,
        0.614501f, 0.537184f,
        0.795655f, 0.497283f,

        -0.99458f, -1.14377f,
        -1.12512f, -1.22176f,
        -0.99458f, -1.14377f,
        -1.12512f, -1.22176f,
        -0.99458f, -1.14377f,

        0.944842f, 0.571876f,
        0.711146f, 1.19437f,
        0.944842f, 0.571876f,
        0.711146f, 1.19437f,
        0.944842f, 0.571876f,

        1.09403f, 0.42269f,
        0.807792f, 0.904437f,
        1.09403f, 0.42269f,
        0.807792f, 0.904437f,
        1.09403f, 0.42269f,

        -1.59133f, -1.74051f,
        -1.5117f, -1.60834f,
        -1.59133f, -1.74051f,
        -1.5117f, -1.60834f,
        -1.59133f, -1.74051f
    };

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("batch_norm").get_memory();
    auto output_ptr = output.pointer<float>();

    std::vector<float> mean_ref = { -3.3333f, -0.3583f, -3.3333f, -0.3583f, -3.3333f };
    std::vector<float> val_ref = { 44.9305f, 107.0624f, 44.9305f, 107.0624f, 44.9305f };

    for (int j = 0; j < 5; ++j) { //F
        float sum = 0, var = 0;

        auto scalep = scale.pointer<float>();
        auto shiftp = shift.pointer<float>();
        float scalef = scalep[j];
        float shiftf = shiftp[j];

        auto meanp = mean_out.pointer<float>();
        auto varp = variance_out.pointer<float>();
        float meanf = meanp[j];
        float varf = varp[j];

        for (int i = 0; i < 2; ++i) { //B
            for (int k = 0; k < 2; ++k) { //Y
                for (int l = 0; l < 3; ++l) { //X
                    int index = 30 * k + 10 * l + 2 * j + i;
                    float data = output_ptr[index];
                    data = (data - shiftf) / scalef;
                    EXPECT_NEAR(data, expected_result[index], 1e-3F);
                    sum += data;
                    var += data * data;
                }
            }
        }
        sum /= 2 * 3 * 2;
        var /= 2 * 3 * 2;

        EXPECT_NEAR(sum, 0, 1e-03F);
        EXPECT_NEAR(var, 1, 1e-03F);

        EXPECT_NEAR(meanf, mean_ref[j], 1e-03F);
        EXPECT_NEAR(varf, val_ref[j], 1e-03F);
    }
}

TEST(batch_normalization_gpu, basic_in2x2x3x5_byxf_scale_shift_different_shapes) {
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::byxf,{ 2, 5, 3, 2 } });
    auto mean = memory::allocate(engine, { data_types::f32, format::byxf,{ 5, 1, 1, 1 } });
    auto variance = memory::allocate(engine, { data_types::f32, format::byxf,{ 1, 5, 1, 1 } });
    auto scale = memory::allocate(engine, { data_types::f32, format::byxf,{ 1, 1, 5, 1 } });
    auto shift = memory::allocate(engine, { data_types::f32, format::byxf,{ 1, 1, 1, 5 } });

    float epsilon = 0.0001f;

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(data("mean", mean));
    topology.add(data("variance", variance));
    topology.add(data("scale", scale));
    topology.add(data("shift", shift));
    topology.add(batch_norm("batch_norm", "input", "mean", "variance", "scale", "shift", epsilon));

    set_values(input, {
        // b0y0
        1.f, 5.f, 1.f, 5.f, 1.f, // x0
        2.f, 6.f, 2.f, 6.f, 2.f, // x1
        -10.f, -12.f, -10.f, -12.f, -10.f, //x2

        // b0y1
        3.f, 7.f, 3.f, 7.f, 3.f,
        4.f, 8.f, 4.f, 8.f, 4.f,
        -14.f, -16.f, -14.f, -16.f, -14.f,
        
        // b1y0
        0.f, 1.5f, 0.f, 1.5f, 0.f,
        0.f, 5.2f, 0.f, 5.2f, 0.f,
        -11.f, -13.f, -11.f, -13.f, -11.f,
        
        // b1y1
        0.5f, 12.f, 0.5f, 12.f, 0.5f,
        -0.5f, 9.f, -0.5f, 9.f, -0.5f,
        -15.f, -17.f, -15.f, -17.f, -15.f
    });

    set_values(mean, { -3.3333f, -0.3583f, -3.3333f, -0.3583f, -3.3333f });
    set_values(variance, { 44.9305f, 107.0624f, 44.9305f, 107.0624f, 44.9305f });
    set_values(scale, { 2.f, 1.f, 3.f, 4.f, 5.f });
    set_values(shift, { 0.f, 5.f, -5.f, -15.f, 0.5f });

    std::vector<float> expected_result{
        0.646469f, 0.517855f, 0.646469f, 0.517855f, 0.646469f,
        0.795655f, 0.614501f, 0.795655f, 0.614501f, 0.795655f,
        -0.99458f, -1.12512f, -0.99458f, -1.12512f, -0.99458f,

        0.944842f, 0.711146f, 0.944842f, 0.711146f, 0.944842f,
        1.09403f, 0.807792f, 1.09403f, 0.807792f, 1.09403f,
        -1.59133f, -1.5117f, -1.59133f, -1.5117f, -1.59133f,

        0.497283f, 0.179596f, 0.497283f, 0.179596f, 0.497283f,
        0.497283f, 0.537184f, 0.497283f, 0.537184f, 0.497283f,
        -1.14377f, -1.22176f, -1.14377f, -1.22176f, -1.14377f,

        0.571876f, 1.19437f, 0.571876f, 1.19437f, 0.571876f,
        0.42269f, 0.904437f, 0.42269f, 0.904437f, 0.42269f,
        -1.74051f, -1.60834f, -1.74051f, -1.60834f, -1.74051f
    };

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("batch_norm").get_memory();
    auto output_ptr = output.pointer<float>();

    for (int j = 0; j < 5; ++j) { //F
        float sum = 0, var = 0;

        auto scalep = scale.pointer<float>();
        auto shiftp = shift.pointer<float>();
        float scalef = scalep[j];
        float shiftf = shiftp[j];

        for (int i = 0; i < 2; ++i) { //B
            for (int k = 0; k < 2; ++k) { //Y
                for (int l = 0; l < 3; ++l) { //X
                    auto index = 30 * i + 15 * k + 5 * l + j;
                    float data = output_ptr[index];
                    data = (data - shiftf) / scalef;
                    EXPECT_NEAR(data, expected_result[index], 1e-3F);
                    sum += data;
                    var += data * data;
                }
            }
        }
        sum /= 2 * 3 * 2;
        var /= 2 * 3 * 2;

        EXPECT_NEAR(sum, 0, 1e-03F);
        EXPECT_NEAR(var, 1, 1e-03F);
    }
}

TEST(batch_normalization_gpu, basic_in2x2x3x5_byxf_with_var_mean_outputs_no_inv_var_different_shapes) {
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::byxf,{ 2, 5, 3, 2 } });
    auto mean_out = memory::allocate(engine, { data_types::f32, format::byxf,{ 5, 1, 1, 1 } });
    auto variance_out = memory::allocate(engine, { data_types::f32, format::byxf,{ 1, 5, 1, 1 } });
    auto scale = memory::allocate(engine, { data_types::f32, format::byxf,{ 1, 1, 5, 1 } });
    auto shift = memory::allocate(engine, { data_types::f32, format::byxf,{ 1, 1, 1, 5 } });

    float epsilon = 0.0001f;

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(data("scale", scale));
    topology.add(data("shift", shift));
    topology.add(mutable_data("mean_out", mean_out));
    topology.add(mutable_data("variance_out", variance_out));
    topology.add(batch_norm("batch_norm", "input", epsilon, "mean_out", "variance_out", "scale", "shift"));

    set_values(input, {
        // b0y0
        1.f, 5.f, 1.f, 5.f, 1.f, // x0
        2.f, 6.f, 2.f, 6.f, 2.f, // x1
        -10.f, -12.f, -10.f, -12.f, -10.f, //x2

        // b0y1
        3.f, 7.f, 3.f, 7.f, 3.f,
        4.f, 8.f, 4.f, 8.f, 4.f,
        -14.f, -16.f, -14.f, -16.f, -14.f,

        // b1y0
        0.f, 1.5f, 0.f, 1.5f, 0.f,
        0.f, 5.2f, 0.f, 5.2f, 0.f,
        -11.f, -13.f, -11.f, -13.f, -11.f,

        // b1y1
        0.5f, 12.f, 0.5f, 12.f, 0.5f,
        -0.5f, 9.f, -0.5f, 9.f, -0.5f,
        -15.f, -17.f, -15.f, -17.f, -15.f
    });

    set_values(scale, { 2.f, 1.f, 3.f, 4.f, 5.f });
    set_values(shift, { 0.f, 5.f, -5.f, -15.f, 0.5f });

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("batch_norm").get_memory();
    auto output_ptr = output.pointer<float>();

    std::vector<float> mean_ref = { -3.3333f, -0.3583f, -3.3333f, -0.3583f, -3.3333f };
    std::vector<float> val_ref = { 44.9305f, 107.0624f, 44.9305f, 107.0624f, 44.9305f };

    std::vector<float> expected_result{
        0.646469f, 0.517855f, 0.646469f, 0.517855f, 0.646469f,
        0.795655f, 0.614501f, 0.795655f, 0.614501f, 0.795655f,
        -0.99458f, -1.12512f, -0.99458f, -1.12512f, -0.99458f,

        0.944842f, 0.711146f, 0.944842f, 0.711146f, 0.944842f,
        1.09403f, 0.807792f, 1.09403f, 0.807792f, 1.09403f,
        -1.59133f, -1.5117f, -1.59133f, -1.5117f, -1.59133f,

        0.497283f, 0.179596f, 0.497283f, 0.179596f, 0.497283f,
        0.497283f, 0.537184f, 0.497283f, 0.537184f, 0.497283f,
        -1.14377f, -1.22176f, -1.14377f, -1.22176f, -1.14377f,

        0.571876f, 1.19437f, 0.571876f, 1.19437f, 0.571876f,
        0.42269f, 0.904437f, 0.42269f, 0.904437f, 0.42269f,
        -1.74051f, -1.60834f, -1.74051f, -1.60834f, -1.74051f
    };

    for (int j = 0; j < 5; ++j) { //F
        float sum = 0, var = 0;

        auto scalep = scale.pointer<float>();
        auto shiftp = shift.pointer<float>();
        float scalef = scalep[j];
        float shiftf = shiftp[j];

        auto meanp = mean_out.pointer<float>();
        auto varp = variance_out.pointer<float>();
        float meanf = meanp[j];
        float varf = varp[j];

        for (int i = 0; i < 2; ++i) { //B
            for (int k = 0; k < 2; ++k) { //Y
                for (int l = 0; l < 3; ++l) { //X
                    auto index = 30 * i + 15 * k + 5 * l + j;
                    float data = output_ptr[index];
                    data = (data - shiftf) / scalef;
                    EXPECT_NEAR(data, expected_result[index], 1e-3F);
                    sum += data;
                    var += data * data;
                }
            }
        }
        sum /= 2 * 3 * 2;
        var /= 2 * 3 * 2;

        EXPECT_NEAR(sum, 0, 1e-03F);
        EXPECT_NEAR(var, 1, 1e-03F);

        EXPECT_NEAR(meanf, mean_ref[j], 1e-03F);
        EXPECT_NEAR(varf, val_ref[j], 1e-03F);
    }
}

TEST(ngraph_batch_normalization_gpu, batchnorm_fprop_b1c2h2w2)
{
    const auto& engine = get_test_engine();

    tensor input_shape{ 1, 2, 2, 2 };
    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, input_shape });
    tensor mean_shape{ feature(2) };
    auto mean = memory::allocate(engine, { data_types::f32, format::bfyx, mean_shape });
    tensor var_shape{ feature(2) };
    auto variance = memory::allocate(engine, { data_types::f32, format::bfyx, var_shape });
    tensor gamma_shape{ feature(2) };
    auto gamma = memory::allocate(engine, { data_types::f32, format::bfyx, gamma_shape });
    tensor beta_shape{ feature(2) };
    auto beta = memory::allocate(engine, { data_types::f32, format::bfyx, beta_shape });

    float eps = 0.001f;

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(data("gamma", gamma));
    topology.add(data("beta", beta));
    topology.add(mutable_data("mean", mean));
    topology.add(mutable_data("variance", variance));
    topology.add(batch_norm("batch_norm", "input", eps, "mean", "variance", "gamma", "beta"));

    set_values<float>(input, {
        0.54881352f,
        0.71518934f,
        0.60276335f,
        0.54488319f,

        0.42365479f,
        0.64589411f,
        0.4375872f,
        0.89177299f
    });

    set_values<float>(gamma, { 1.f, 1.f });
    set_values<float>(beta, { 0.f, 0.f });

    std::vector<float> expected_result { 
        -0.71498716f,
        1.48388731f,
        -0.00196938f,
        -0.76693159f,

        -0.91316032f,
        0.23943391f,
        -0.84090298f,
        1.51462936f 
    };

    std::vector<float> expected_mean = { 0.602912f, 0.599727f };
    std::vector<float> expected_variance = { 0.00472505f, 0.0361782f };

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("batch_norm").get_memory();
    auto output_ptr = output.pointer<float>();

    for (int j = 0; j < 2; ++j) { //F
        float sum = 0;

        auto scalep = gamma.pointer<float>();
        auto shiftp = beta.pointer<float>();
        float scalef = scalep[j];
        float shiftf = shiftp[j];

        auto meanp = mean.pointer<float>();
        auto varp = variance.pointer<float>();
        float meanf = meanp[j];
        float varf = varp[j];

        for (int k = 0; k < 2; ++k) { //Y
            for (int l = 0; l < 2; ++l) { //X
                int index = 4 * j + 2 * k + l;
                float data = output_ptr[index];
                data = (data - shiftf) / scalef;
                EXPECT_NEAR(data, expected_result[index], 1e-5F);
                sum += data;
            }
        }

        sum /= 2 * 2;

        EXPECT_NEAR(sum, 0, 1e-5F);

        EXPECT_NEAR(meanf, expected_mean[j], 1e-5F);
        EXPECT_NEAR(varf, expected_variance[j], 1e-5F);
    }
}

TEST(ngraph_batch_normalization_gpu, batchnorm_fprop_b2c2h2w1)
{
    const auto& engine = get_test_engine();

    tensor input_shape{ 2, 2, 1, 2 };
    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, input_shape });
    tensor mean_shape{ feature(2) };
    auto mean = memory::allocate(engine, { data_types::f32, format::bfyx, mean_shape });
    tensor var_shape{ feature(2) };
    auto variance = memory::allocate(engine, { data_types::f32, format::bfyx, var_shape });
    tensor gamma_shape{ feature(2) };
    auto gamma = memory::allocate(engine, { data_types::f32, format::bfyx, gamma_shape });
    tensor beta_shape{ feature(2) };
    auto beta = memory::allocate(engine, { data_types::f32, format::bfyx, beta_shape });

    float eps = 0.001f;

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(data("gamma", gamma));
    topology.add(data("beta", beta));
    topology.add(mutable_data("mean", mean));
    topology.add(mutable_data("variance", variance));
    topology.add(batch_norm("batch_norm", "input", eps, "mean", "variance", "gamma", "beta"));

    set_values<float>(input, { 
        0.54881352f,
        0.71518934f,

        0.60276335f,
        0.54488319f,

        0.42365479f,
        0.64589411f,

        0.4375872f,
        0.89177299f
    });

    set_values<float>(gamma, { 1.f, 1.f });
    set_values<float>(beta, { 0.f, 0.f });

    std::vector<float> expected_result{
        -0.30327f, 
        1.1561f, 

        -0.0963782f, 
        -0.434702f, 
        

        -1.4011f, 
        0.548275f, 

        -1.06187f,
        1.59295f };

    std::vector<float> expected_mean = { 0.583388f, 0.619252f };
    std::vector<float> expected_variance = { 0.0119972f, 0.0282681f };
    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("batch_norm").get_memory();
    auto output_ptr = output.pointer<float>();

    for (int j = 0; j < 2; ++j) { //F
        float sum = 0;

        auto scalep = gamma.pointer<float>();
        auto shiftp = beta.pointer<float>();
        float scalef = scalep[j];
        float shiftf = shiftp[j];

        auto meanp = mean.pointer<float>();
        auto varp = variance.pointer<float>();
        float meanf = meanp[j];
        float varf = varp[j];

        for (int k = 0; k < 2; ++k) { //B
            for (int l = 0; l < 2; ++l) { //Y
                int index = 4 * k + 2 * j + l;
                float data = output_ptr[index];
                data = (data - shiftf) / scalef;
                EXPECT_NEAR(data, expected_result[index], 1e-5F);
                sum += data;
            }
        }

        sum /= 2 * 2;

        EXPECT_NEAR(sum, 0, 1e-5F);

        EXPECT_NEAR(meanf, expected_mean[j], 1e-5F);
        EXPECT_NEAR(varf, expected_variance[j], 1e-5F);
    }
}

TEST(ngraph_batch_normalization_gpu, batchnorm_fprop_inference_b2c2h2w1)
{
    const auto& engine = get_test_engine();

    tensor input_shape{ 2, 2, 1, 2 };
    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, input_shape });
    tensor mean_shape{ feature(2) };
    auto mean = memory::allocate(engine, { data_types::f32, format::bfyx, mean_shape });
    tensor var_shape{ feature(2) };
    auto variance = memory::allocate(engine, { data_types::f32, format::bfyx, var_shape });
    tensor gamma_shape{ feature(2) };
    auto gamma = memory::allocate(engine, { data_types::f32, format::bfyx, gamma_shape });
    tensor beta_shape{ feature(2) };
    auto beta = memory::allocate(engine, { data_types::f32, format::bfyx, beta_shape });

    float eps = 0.001f;

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(data("gamma", gamma));
    topology.add(data("beta", beta));
    topology.add(data("mean", mean));
    topology.add(data("variance", variance));
    topology.add(batch_norm("batch_norm", "input", eps, "mean", "variance", "gamma", "beta"));

    set_values<float>(input, { 
        0.54881352f,
        0.71518934f,

        0.60276335f,
        0.54488319f,

        0.42365479f,
        0.64589411f,

        0.4375872f,
        0.89177299f
    });

    set_values<float>(gamma, { 1.f, 1.f });
    set_values<float>(beta, { 0.f, 0.f });

    set_values<float>(mean, { 0.583388f, 0.619252f });
    set_values<float>(variance, { 0.0119972f, 0.0282681f });

    std::vector<float> expected_result{
        -0.30327f,
        1.1561f,

        -0.0963782f,
        -0.434702f,
        
        
        -1.4011f,
        0.548275f,
        
        -1.06187f,
        1.59295f };

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("batch_norm").get_memory();
    auto output_ptr = output.pointer<float>();

    for (int j = 0; j < 2; ++j) { //F
        float sum = 0;

        auto scalep = gamma.pointer<float>();
        auto shiftp = beta.pointer<float>();
        float scalef = scalep[j];
        float shiftf = shiftp[j];

        for (int k = 0; k < 2; ++k) { //B
            for (int l = 0; l < 2; ++l) { //Y
                int index = 4 * k + 2 * j + l;
                float data = output_ptr[index];
                data = (data - shiftf) / scalef;
                EXPECT_NEAR(data, expected_result[index], 1e-5F);
                sum += data;
            }
        }

        sum /= 2 * 2;

        EXPECT_NEAR(sum, 0, 1e-5F);
    }
}

TEST(ngraph_batch_normalization_gpu, batchnorm_fprop_b2c2h2w1_different_shapes)
{
    const auto& engine = get_test_engine();

    tensor input_shape = { 2, 2, 1, 2 };
    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, input_shape });
    tensor mean_shape = { 2, 1, 1, 1 };
    auto mean = memory::allocate(engine, { data_types::f32, format::bfyx, mean_shape });
    tensor var_shape = { 1, 2, 1, 1 };
    auto variance = memory::allocate(engine, { data_types::f32, format::bfyx, var_shape });
    tensor gamma_shape = { 1, 1, 2, 1 };
    auto gamma = memory::allocate(engine, { data_types::f32, format::bfyx, gamma_shape });
    tensor beta_shape = { 1, 1, 1, 2 };
    auto beta = memory::allocate(engine, { data_types::f32, format::bfyx, beta_shape });

    float eps = 0.001f;

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(data("gamma", gamma));
    topology.add(data("beta", beta));
    topology.add(mutable_data("mean", mean));
    topology.add(mutable_data("variance", variance));
    topology.add(batch_norm("batch_norm", "input", eps, "mean", "variance", "gamma", "beta"));

    set_values<float>(input, {
        0.54881352f,
        0.71518934f,

        0.60276335f,
        0.54488319f,

        0.42365479f,
        0.64589411f,

        0.4375872f,
        0.89177299f
    });

    set_values<float>(gamma, { 2.f, 3.f });
    set_values<float>(beta, { 5.f, 10.f });

    std::vector<float> expected_result{
        -0.30327f,
        1.1561f,

        -0.0963782f,
        -0.434702f,

        -1.4011f,
        0.548275f,

        -1.06187f,
        1.59295f };

    std::vector<float> expected_mean = { 0.583388f, 0.619252f };
    std::vector<float> expected_variance = { 0.0119972f, 0.0282681f };
    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("batch_norm").get_memory();
    auto output_ptr = output.pointer<float>();

    for (int j = 0; j < 2; ++j) { //F
        float sum = 0;

        auto scalep = gamma.pointer<float>();
        auto shiftp = beta.pointer<float>();
        float scalef = scalep[j];
        float shiftf = shiftp[j];

        auto meanp = mean.pointer<float>();
        auto varp = variance.pointer<float>();
        float meanf = meanp[j];
        float varf = varp[j];

        for (int k = 0; k < 2; ++k) { //B
            for (int l = 0; l < 2; ++l) { //Y
                int index = 4 * k + 2 * j + l;
                float data = output_ptr[index];
                data = (data - shiftf) / scalef;
                EXPECT_NEAR(data, expected_result[index], 1e-5F);
                sum += data;
            }
        }

        sum /= 2 * 2;

        EXPECT_NEAR(sum, 0, 1e-5F);

        EXPECT_NEAR(meanf, expected_mean[j], 1e-5F);
        EXPECT_NEAR(varf, expected_variance[j], 1e-5F);
    }
}

TEST(ngraph_batch_normalization_gpu, batchnorm_fprop_inference_b2c2h2w1_different_shapes)
{
    const auto& engine = get_test_engine();

    tensor input_shape = { 2, 2, 1, 2 };
    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, input_shape });
    tensor mean_shape = { 2, 1, 1, 1 };
    auto mean = memory::allocate(engine, { data_types::f32, format::bfyx, mean_shape });
    tensor var_shape = { 1, 1, 2, 1 };
    auto variance = memory::allocate(engine, { data_types::f32, format::bfyx, var_shape });
    tensor gamma_shape = { 1, 1, 2, 1 };
    auto gamma = memory::allocate(engine, { data_types::f32, format::bfyx, gamma_shape });
    tensor beta_shape = { 1, 1, 1, 2 };
    auto beta = memory::allocate(engine, { data_types::f32, format::bfyx, beta_shape });

    float eps = 0.001f;

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(data("gamma", gamma));
    topology.add(data("beta", beta));
    topology.add(data("mean", mean));
    topology.add(data("variance", variance));
    topology.add(batch_norm("batch_norm", "input", eps, "mean", "variance", "gamma", "beta"));

    set_values<float>(input, {
        0.54881352f,
        0.71518934f,

        0.60276335f,
        0.54488319f,

        0.42365479f,
        0.64589411f,

        0.4375872f,
        0.89177299f
    });

    set_values<float>(gamma, { 2.f, 3.f });
    set_values<float>(beta, { 5.f, 10.f });

    set_values<float>(mean, { 0.583388f, 0.619252f });
    set_values<float>(variance, { 0.0119972f, 0.0282681f });

    std::vector<float> expected_result{
        -0.30327f,
        1.1561f,

        -0.0963782f,
        -0.434702f,

        -1.4011f,
        0.548275f,

        -1.06187f,
        1.59295f };

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("batch_norm").get_memory();
    auto output_ptr = output.pointer<float>();

    for (int j = 0; j < 2; ++j) { //F
        float sum = 0;

        auto scalep = gamma.pointer<float>();
        auto shiftp = beta.pointer<float>();
        float scalef = scalep[j];
        float shiftf = shiftp[j];

        for (int k = 0; k < 2; ++k) { //B
            for (int l = 0; l < 2; ++l) { //Y
                int index = 4 * k + 2 * j + l;
                float data = output_ptr[index];
                data = (data - shiftf) / scalef;
                EXPECT_NEAR(data, expected_result[index], 1e-5F);
                sum += data;
            }
        }

        sum /= 2 * 2;

        EXPECT_NEAR(sum, 0, 1e-5F);
    }
}

TEST(ngraph_batch_normalization_gpu, batchnorm_fprop_b2c5h2w1_different_shapes)
{
    const auto& engine = get_test_engine();

    tensor input_shape = { 2, 5, 1, 2 };
    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, input_shape });
    tensor mean_shape = { 5, 1, 1, 1 };
    auto mean = memory::allocate(engine, { data_types::f32, format::bfyx, mean_shape });
    tensor var_shape = { 1, 5, 1, 1 };
    auto variance = memory::allocate(engine, { data_types::f32, format::bfyx, var_shape });
    tensor gamma_shape = { 1, 1, 5, 1 };
    auto gamma = memory::allocate(engine, { data_types::f32, format::bfyx, gamma_shape });
    tensor beta_shape = { 1, 1, 1, 5 };
    auto beta = memory::allocate(engine, { data_types::f32, format::bfyx, beta_shape });

    float eps = 0.001f;

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(data("gamma", gamma));
    topology.add(data("beta", beta));
    topology.add(mutable_data("mean", mean));
    topology.add(mutable_data("variance", variance));
    topology.add(batch_norm("batch_norm", "input", eps, "mean", "variance", "gamma", "beta"));

    set_values<float>(input, {
        0.54881352f,
        0.71518934f,

        0.60276335f,
        0.54488319f,

        0.54881352f,
        0.71518934f,

        0.60276335f,
        0.54488319f,

        0.54881352f,
        0.71518934f,

        0.42365479f,
        0.64589411f,

        0.4375872f,
        0.89177299f,

        0.42365479f,
        0.64589411f,

        0.4375872f,
        0.89177299f,

        0.42365479f,
        0.64589411f
    });

    set_values<float>(gamma, { 2.f, 3.f, 4.f, 5.f, 1.f });
    set_values<float>(beta, { 5.f, 10.f, -10.f, -15.f, 0.f });

    std::vector<float> expected_result{
        -0.30327f,
        1.1561f,

        -0.0963782f,
        -0.434702f,

        -0.30327f,
        1.1561f,

        -0.0963782f,
        -0.434702f,

        -0.30327f,
        1.1561f,

        -1.4011f,
        0.548275f,

        -1.06187f,
        1.59295f,

        -1.4011f,
        0.548275f,

        -1.06187f,
        1.59295f,

        -1.4011f,
        0.548275f
    };

    std::vector<float> expected_mean = { 0.583388f, 0.619252f, 0.583388f, 0.619252f, 0.583388f };
    std::vector<float> expected_variance = { 0.0119972f, 0.0282681f, 0.0119972f, 0.0282681f, 0.0119972f };
    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("batch_norm").get_memory();
    auto output_ptr = output.pointer<float>();

    for (int j = 0; j < 5; ++j) { //F
        float sum = 0;

        auto scalep = gamma.pointer<float>();
        auto shiftp = beta.pointer<float>();
        float scalef = scalep[j];
        float shiftf = shiftp[j];

        auto meanp = mean.pointer<float>();
        auto varp = variance.pointer<float>();
        float meanf = meanp[j];
        float varf = varp[j];

        for (int k = 0; k < 2; ++k) { //B
            for (int l = 0; l < 2; ++l) { //Y
                int index = 10 * k + 2 * j + l;
                float data = output_ptr[index];
                data = (data - shiftf) / scalef;
                EXPECT_NEAR(data, expected_result[index], 1e-5F);
                sum += data;
            }
        }

        sum /= 2 * 2;

        EXPECT_NEAR(sum, 0, 1e-5F);

        EXPECT_NEAR(meanf, expected_mean[j], 1e-5F);
        EXPECT_NEAR(varf, expected_variance[j], 1e-5F);
    }
}

TEST(ngraph_batch_normalization_gpu, batchnorm_fprop_inference_b2c5h2w1_different_shapes)
{
    const auto& engine = get_test_engine();

    tensor input_shape = { 2, 5, 1, 2 };
    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, input_shape });
    tensor mean_shape = { 5, 1, 1, 1 };
    auto mean = memory::allocate(engine, { data_types::f32, format::bfyx, mean_shape });
    tensor var_shape = { 1, 5, 1, 1 };
    auto variance = memory::allocate(engine, { data_types::f32, format::bfyx, var_shape });
    tensor gamma_shape = { 1, 1, 5, 1 };
    auto gamma = memory::allocate(engine, { data_types::f32, format::bfyx, gamma_shape });
    tensor beta_shape = { 1, 1, 1, 5 };
    auto beta = memory::allocate(engine, { data_types::f32, format::bfyx, beta_shape });

    float eps = 0.001f;

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(data("gamma", gamma));
    topology.add(data("beta", beta));
    topology.add(data("mean", mean));
    topology.add(data("variance", variance));
    topology.add(batch_norm("batch_norm", "input", eps, "mean", "variance", "gamma", "beta"));

    set_values<float>(input, {
        0.54881352f,
        0.71518934f,

        0.60276335f,
        0.54488319f,

        0.54881352f,
        0.71518934f,

        0.60276335f,
        0.54488319f,

        0.54881352f,
        0.71518934f,

        0.42365479f,
        0.64589411f,

        0.4375872f,
        0.89177299f,

        0.42365479f,
        0.64589411f,

        0.4375872f,
        0.89177299f,

        0.42365479f,
        0.64589411f
    });

    set_values<float>(gamma, { 2.f, 3.f, 4.f, 5.f, 1.f });
    set_values<float>(beta, { 5.f, 10.f, -10.f, -15.f, 0.f });

    std::vector<float> expected_result{
        -0.30327f,
        1.1561f,

        -0.0963782f,
        -0.434702f,

        -0.30327f,
        1.1561f,

        -0.0963782f,
        -0.434702f,

        -0.30327f,
        1.1561f,

        -1.4011f,
        0.548275f,

        -1.06187f,
        1.59295f,

        -1.4011f,
        0.548275f,

        -1.06187f,
        1.59295f,

        -1.4011f,
        0.548275f
    };

    set_values<float>(mean, { 0.583388f, 0.619252f, 0.583388f, 0.619252f, 0.583388f });
    set_values<float>(variance, { 0.0119972f, 0.0282681f, 0.0119972f, 0.0282681f, 0.0119972f });
    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("batch_norm").get_memory();
    auto output_ptr = output.pointer<float>();

    for (int j = 0; j < 5; ++j) { //F
        float sum = 0;

        auto scalep = gamma.pointer<float>();
        auto shiftp = beta.pointer<float>();
        float scalef = scalep[j];
        float shiftf = shiftp[j];

        for (int k = 0; k < 2; ++k) { //B
            for (int l = 0; l < 2; ++l) { //Y
                int index = 10 * k + 2 * j + l;
                float data = output_ptr[index];
                data = (data - shiftf) / scalef;
                EXPECT_NEAR(data, expected_result[index], 1e-5F);
                sum += data;
            }
        }

        sum /= 2 * 2;

        EXPECT_NEAR(sum, 0, 1e-5F);
    }
}

TEST(ngraph_batch_normalization_gpu, batchnorm_fprop_b1c2h2w2_no_bn_output)
{
    engine engine;

    tensor input_shape{ 1, 2, 2, 2 };
    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, input_shape });
    tensor mean_shape{ feature(2) };
    auto mean = memory::allocate(engine, { data_types::f32, format::bfyx, mean_shape });
    tensor var_shape{ feature(2) };
    auto variance = memory::allocate(engine, { data_types::f32, format::bfyx, var_shape });
    tensor gamma_shape{ feature(2) };
    auto gamma = memory::allocate(engine, { data_types::f32, format::bfyx, gamma_shape });
    tensor beta_shape{ feature(2) };
    auto beta = memory::allocate(engine, { data_types::f32, format::bfyx, beta_shape });

    float eps = 0.001f;

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(data("gamma", gamma));
    topology.add(data("beta", beta));
    topology.add(mutable_data("mean", mean));
    topology.add(mutable_data("variance", variance));
    topology.add(batch_norm("batch_norm", "input", eps, "mean", "variance", "gamma", "beta"));

    set_values<float>(input, {
        0.54881352f,
        0.71518934f,
        0.60276335f,
        0.54488319f,

        0.42365479f,
        0.64589411f,
        0.4375872f,
        0.89177299f
    });

    set_values<float>(gamma, { 1.f, 1.f });
    set_values<float>(beta, { 0.f, 0.f });

    std::vector<float> expected_mean = { 0.602912f, 0.599727f };
    std::vector<float> expected_variance = { 0.00472505f, 0.0361782f };

    build_options bo;
    bo.set_option(build_option::outputs({ "mean", "variance" }));
    network network(engine, topology, bo);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    for (int j = 0; j < 2; ++j) { //F
        auto meanp = mean.pointer<float>();
        auto varp = variance.pointer<float>();
        float meanf = meanp[j];
        float varf = varp[j];

        EXPECT_NEAR(meanf, expected_mean[j], 1e-5F);
        EXPECT_NEAR(varf, expected_variance[j], 1e-5F);
    }
}
