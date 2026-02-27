// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/primitives/gated_mlp.hpp"
#include "intel_gpu/primitives/input_layout.hpp"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "network_test.h"
#include "test_utils.h"
#include <cmath>
#include <string>
#include <vector>

#ifdef ENABLE_ONEDNN_FOR_GPU

using namespace cldnn;
using namespace ::tests;

namespace {

float swish(float x) {
	return x / (1.0f + std::exp(-x));
}

std::vector<ov::float16> to_f16(const std::vector<float>& src) {
	std::vector<ov::float16> dst;
	dst.reserve(src.size());
	for (auto v : src) {
		dst.push_back(ov::float16(v));
	}
	return dst;
}

std::vector<float> gated_mlp_reference(const std::vector<float>& src,
									   const std::vector<float>& w_gate,
									   const std::vector<float>& w_up,
									   const std::vector<float>& w_down,
									   int batch,
									   int ifm,
									   int hidden) {
	std::vector<float> gate(batch * hidden, 0.0f);
	std::vector<float> up(batch * hidden, 0.0f);
	std::vector<float> out(batch * ifm, 0.0f);

	for (int b = 0; b < batch; b++) {
		for (int h = 0; h < hidden; h++) {
			float gate_acc = 0.0f;
			float up_acc = 0.0f;
			for (int i = 0; i < ifm; i++) {
				gate_acc += src[b * ifm + i] * w_gate[i * hidden + h];
				up_acc += src[b * ifm + i] * w_up[i * hidden + h];
			}
			gate[b * hidden + h] = swish(gate_acc);
			up[b * hidden + h] = up_acc;
		}
	}

	for (int b = 0; b < batch; b++) {
		for (int i = 0; i < ifm; i++) {
			float acc = 0.0f;
			for (int h = 0; h < hidden; h++) {
				acc += (gate[b * hidden + h] * up[b * hidden + h]) * w_down[h * ifm + i];
			}
			out[b * ifm + i] = acc;
		}
	}

	return out;
}

template <typename QType, typename ZpType>
std::vector<float> dequantize_weights(const std::vector<QType>& qweights,
									  int rows,
									  int cols,
									  const std::vector<float>& scales,
									  int scale_rows,
									  int scale_cols,
									  const std::vector<ZpType>& zps,
									  int zp_rows,
									  int zp_cols) {
	std::vector<float> weights(rows * cols, 0.0f);
	const int scale_group_size = rows / scale_rows;
	const int zp_group_size = rows / zp_rows;

	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			const int scale_row = r / scale_group_size;
			const int scale_col = (scale_cols == 1 ? 0 : c);
			const int zp_row = r / zp_group_size;
			const int zp_col = (zp_cols == 1 ? 0 : c);

			const float scale = scales[scale_row * scale_cols + scale_col];
			const int zp = static_cast<int>(zps[zp_row * zp_cols + zp_col]);
			const int q = static_cast<int>(qweights[r * cols + c]);
			weights[r * cols + c] = static_cast<float>(q - zp) * scale;
		}
	}

	return weights;
}

void check_output(memory::ptr output_mem, const std::vector<float>& ref, float tol = 1e-2f) {
	cldnn::mem_lock<ov::float16> output_ptr(output_mem, get_test_stream());
	ASSERT_EQ(output_ptr.size(), ref.size());

	for (size_t i = 0; i < ref.size(); i++) {
		ASSERT_NEAR(static_cast<float>(output_ptr[i]), ref[i], tol) << "idx=" << i;
	}
}

void check_output_finite(memory::ptr output_mem) {
	cldnn::mem_lock<ov::float16> output_ptr(output_mem, get_test_stream());
	for (size_t i = 0; i < output_ptr.size(); ++i) {
		ASSERT_TRUE(std::isfinite(static_cast<float>(output_ptr[i]))) << "idx=" << i;
	}
}

void skip_if_no_immad(engine& engine) {
	if (!engine.get_device_info().supports_immad) {
		GTEST_SKIP() << "gated_mlp oneDNN implementation requires IMMAD support";
	}
}

std::shared_ptr<network> create_fp16_network(engine& engine,
											 int batch,
											 int ifm,
											 int hidden,
											 memory::ptr src,
											 memory::ptr w_gate,
											 memory::ptr w_up,
											 memory::ptr w_down) {
	topology topology(
		input_layout("src", layout({batch, 1, 1, ifm}, data_types::f16, format::bfyx)),
		data("w_gate", w_gate),
		data("w_up", w_up),
		data("w_down", w_down),
		reorder("src_2d", input_info("src"), {data_types::f16, format::bfyx, tensor(batch, ifm, 1, 1)}),
		gated_mlp("gmlp",
				  input_info("src_2d"),
				  input_info("w_gate"),
				  input_info("w_up"),
				  input_info("w_down"),
				  ov::op::internal::GLU::GluType::Swish,
				  tensor(batch, ifm, 1, 1),
				  data_types::f16)
	);

	auto config = get_test_default_config(engine);
	config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
	config.set_property(ov::intel_gpu::optimize_data(true));
	config.set_property(ov::intel_gpu::use_onednn(true));

	auto net = std::make_shared<network>(engine, topology, config);
	net->set_input_data("src", src);
	return net;
}

std::shared_ptr<network> create_compressed_network(engine& engine,
												   int batch,
												   int ifm,
												   int hidden,
												   memory::ptr src,
												   memory::ptr w_gate,
												   memory::ptr w_up,
												   memory::ptr w_down,
												   memory::ptr s_gate,
												   memory::ptr s_up,
												   memory::ptr s_down,
												   memory::ptr zp_gate,
												   memory::ptr zp_up,
													memory::ptr zp_down,
													const std::string& prim_name = "gmlp") {
	topology topology(
		input_layout("src", layout({batch, 1, 1, ifm}, data_types::f16, format::bfyx)),
		data("w_gate", w_gate),
		data("w_up", w_up),
		data("w_down", w_down),
		data("s_gate", s_gate),
		data("s_up", s_up),
		data("s_down", s_down),
		data("zp_gate", zp_gate),
		data("zp_up", zp_up),
		data("zp_down", zp_down),
		reorder("src_2d", input_info("src"), {data_types::f16, format::bfyx, tensor(batch, ifm, 1, 1)}),
		gated_mlp(prim_name,
				  input_info("src_2d"),
				  input_info("w_gate"),
				  input_info("w_up"),
				  input_info("w_down"),
				  input_info("s_gate"),
				  input_info("s_up"),
				  input_info("s_down"),
				  input_info("zp_gate"),
				  input_info("zp_up"),
				  input_info("zp_down"),
				  ov::op::internal::GLU::GluType::Swish,
				  tensor(batch, ifm, 1, 1),
				  data_types::f16)
	);

	auto config = get_test_default_config(engine);
	config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
	config.set_property(ov::intel_gpu::optimize_data(true));
	config.set_property(ov::intel_gpu::use_onednn(true));

	auto net = std::make_shared<network>(engine, topology, config);
	net->set_input_data("src", src);
	return net;
}

}  // namespace

TEST(gated_mlp_gpu, fp16_basic_small) {
	auto& engine = get_test_engine();
	skip_if_no_immad(engine);
	const int batch = 2;
	const int ifm = 4;
	const int hidden = 3;

	auto src_mem = engine.allocate_memory({{batch, 1, 1, ifm}, data_types::f16, format::bfyx});
	auto gate_mem = engine.allocate_memory({{ifm, hidden}, data_types::f16, format::bfyx});
	auto up_mem = engine.allocate_memory({{ifm, hidden}, data_types::f16, format::bfyx});
	auto down_mem = engine.allocate_memory({{hidden, ifm}, data_types::f16, format::bfyx});

	std::vector<float> src = {0.5f, -0.2f, 1.0f, 0.7f,
							  -0.8f, 0.3f, 0.1f, 0.9f};
	std::vector<float> gate = {0.2f, -0.3f, 0.5f,
							   0.1f, 0.4f, -0.2f,
							   -0.6f, 0.7f, 0.8f,
							   0.9f, -0.1f, 0.3f};
	std::vector<float> up = {0.4f, 0.2f, -0.5f,
							 -0.3f, 0.6f, 0.1f,
							 0.8f, -0.4f, 0.2f,
							 0.7f, 0.5f, -0.6f};
	std::vector<float> down = {0.1f, -0.2f, 0.3f, 0.4f,
							   -0.5f, 0.6f, -0.7f, 0.8f,
							   0.9f, 0.1f, -0.2f, 0.3f};

	set_values(src_mem, to_f16(src));
	set_values(gate_mem, to_f16(gate));
	set_values(up_mem, to_f16(up));
	set_values(down_mem, to_f16(down));

	auto ref = gated_mlp_reference(src, gate, up, down, batch, ifm, hidden);
	auto net = create_fp16_network(engine, batch, ifm, hidden, src_mem, gate_mem, up_mem, down_mem);
	auto outputs = net->execute();

	ASSERT_EQ(outputs.size(), size_t(1));
	ASSERT_EQ(outputs.begin()->first, "gmlp");
	check_output(outputs.at("gmlp").get_memory(), ref);
}

TEST(gated_mlp_gpu, fp16_basic_medium) {
	auto& engine = get_test_engine();
	skip_if_no_immad(engine);
	const int batch = 1;
	const int ifm = 3;
	const int hidden = 4;

	auto src_mem = engine.allocate_memory({{batch, 1, 1, ifm}, data_types::f16, format::bfyx});
	auto gate_mem = engine.allocate_memory({{ifm, hidden}, data_types::f16, format::bfyx});
	auto up_mem = engine.allocate_memory({{ifm, hidden}, data_types::f16, format::bfyx});
	auto down_mem = engine.allocate_memory({{hidden, ifm}, data_types::f16, format::bfyx});

	std::vector<float> src = {1.0f, -1.5f, 0.25f};
	std::vector<float> gate = {0.3f, -0.4f, 0.8f, -0.2f,
							   -0.7f, 0.5f, 0.1f, 0.6f,
							   0.2f, -0.9f, 0.4f, 0.3f};
	std::vector<float> up = {-0.1f, 0.2f, 0.3f, -0.4f,
							 0.5f, 0.6f, -0.7f, 0.8f,
							 0.9f, -0.2f, 0.1f, -0.3f};
	std::vector<float> down = {0.2f, -0.3f, 0.4f,
							   -0.5f, 0.6f, -0.7f,
							   0.8f, 0.1f, -0.2f,
							   -0.4f, 0.5f, 0.6f};

	set_values(src_mem, to_f16(src));
	set_values(gate_mem, to_f16(gate));
	set_values(up_mem, to_f16(up));
	set_values(down_mem, to_f16(down));

	auto ref = gated_mlp_reference(src, gate, up, down, batch, ifm, hidden);
	auto net = create_fp16_network(engine, batch, ifm, hidden, src_mem, gate_mem, up_mem, down_mem);
	auto outputs = net->execute();

	ASSERT_EQ(outputs.size(), size_t(1));
	ASSERT_EQ(outputs.begin()->first, "gmlp");
	check_output(outputs.at("gmlp").get_memory(), ref);
}

TEST(gated_mlp_gpu, compressed_u8_per_tensor_zp) {
	auto& engine = get_test_engine();
	skip_if_no_immad(engine);
	const int batch = 2;
	const int ifm = 4;
	const int hidden = 3;

	auto src_mem = engine.allocate_memory({{batch, 1, 1, ifm}, data_types::f16, format::bfyx});
	auto gate_mem = engine.allocate_memory({{ifm, hidden}, data_types::u8, format::bfyx});
	auto up_mem = engine.allocate_memory({{ifm, hidden}, data_types::u8, format::bfyx});
	auto down_mem = engine.allocate_memory({{hidden, ifm}, data_types::u8, format::bfyx});

	auto s_gate_mem = engine.allocate_memory({{1, 1}, data_types::f16, format::bfyx});
	auto s_up_mem = engine.allocate_memory({{1, 1}, data_types::f16, format::bfyx});
	auto s_down_mem = engine.allocate_memory({{1, 1}, data_types::f16, format::bfyx});

	auto zp_gate_mem = engine.allocate_memory({{1, 1}, data_types::u8, format::bfyx});
	auto zp_up_mem = engine.allocate_memory({{1, 1}, data_types::u8, format::bfyx});
	auto zp_down_mem = engine.allocate_memory({{1, 1}, data_types::u8, format::bfyx});

	std::vector<float> src = {0.25f, -0.5f, 0.75f, 1.0f,
							  -1.0f, 0.5f, 0.0f, 0.2f};
	std::vector<uint8_t> gate_q = {130, 129, 128, 127, 131, 132, 126, 125, 128, 129, 130, 127};
	std::vector<uint8_t> up_q = {129, 128, 127, 130, 131, 126, 125, 124, 129, 130, 128, 127};
	std::vector<uint8_t> down_q = {127, 128, 129, 130, 131, 132, 126, 125, 124, 128, 129, 130};

	std::vector<float> s_gate = {0.10f};
	std::vector<float> s_up = {0.08f};
	std::vector<float> s_down = {0.05f};
	std::vector<uint8_t> zp_gate = {128};
	std::vector<uint8_t> zp_up = {127};
	std::vector<uint8_t> zp_down = {126};

	set_values(src_mem, to_f16(src));
	set_values(gate_mem, gate_q);
	set_values(up_mem, up_q);
	set_values(down_mem, down_q);
	set_values(s_gate_mem, to_f16(s_gate));
	set_values(s_up_mem, to_f16(s_up));
	set_values(s_down_mem, to_f16(s_down));
	set_values(zp_gate_mem, zp_gate);
	set_values(zp_up_mem, zp_up);
	set_values(zp_down_mem, zp_down);

	auto gate = dequantize_weights<uint8_t, uint8_t>(gate_q, ifm, hidden, s_gate, 1, 1, zp_gate, 1, 1);
	auto up = dequantize_weights<uint8_t, uint8_t>(up_q, ifm, hidden, s_up, 1, 1, zp_up, 1, 1);
	auto down = dequantize_weights<uint8_t, uint8_t>(down_q, hidden, ifm, s_down, 1, 1, zp_down, 1, 1);
	auto ref = gated_mlp_reference(src, gate, up, down, batch, ifm, hidden);

	auto net = create_compressed_network(engine, batch, ifm, hidden,
                                         src_mem, gate_mem, up_mem, down_mem, s_gate_mem, s_up_mem, s_down_mem, zp_gate_mem, zp_up_mem, zp_down_mem);
	auto outputs = net->execute();

	ASSERT_EQ(outputs.size(), size_t(1));
	ASSERT_EQ(outputs.begin()->first, "gmlp");
	check_output(outputs.at("gmlp").get_memory(), ref, 2e-2f);
}

TEST(gated_mlp_gpu, compressed_u8_grouped_zp) {
	auto& engine = get_test_engine();
	skip_if_no_immad(engine);
	const int batch = 32;
	const int ifm = 32;
	const int hidden = 32;

	auto src_mem = engine.allocate_memory({{batch, 1, 1, ifm}, data_types::f16, format::bfyx});
	auto gate_mem = engine.allocate_memory({{ifm, hidden}, data_types::u8, format::bfyx});
	auto up_mem = engine.allocate_memory({{ifm, hidden}, data_types::u8, format::bfyx});
	auto down_mem = engine.allocate_memory({{hidden, ifm}, data_types::u8, format::bfyx});

	auto s_gate_mem = engine.allocate_memory({{2, 1}, data_types::f16, format::bfyx});
	auto s_up_mem = engine.allocate_memory({{2, 1}, data_types::f16, format::bfyx});
	auto s_down_mem = engine.allocate_memory({{2, 1}, data_types::f16, format::bfyx});

	auto zp_gate_mem = engine.allocate_memory({{2, 1}, data_types::u8, format::bfyx});
	auto zp_up_mem = engine.allocate_memory({{2, 1}, data_types::u8, format::bfyx});
	auto zp_down_mem = engine.allocate_memory({{2, 1}, data_types::u8, format::bfyx});

	std::vector<float> src(batch * ifm, 0.0f);
	for (size_t i = 0; i < src.size(); i++) {
		src[i] = (i % 5 == 0) ? -0.5f : 0.25f;
	}

	std::vector<uint8_t> gate_q(ifm * hidden, 129);
	std::vector<uint8_t> up_q(ifm * hidden, 127);
	std::vector<uint8_t> down_q(hidden * ifm, 130);
	for (size_t i = 0; i < gate_q.size(); i++) {
		gate_q[i] = static_cast<uint8_t>(126 + (i % 7));
		up_q[i] = static_cast<uint8_t>(124 + (i % 9));
		down_q[i] = static_cast<uint8_t>(123 + (i % 11));
	}

	std::vector<float> s_gate = {0.02f, 0.03f};
	std::vector<float> s_up = {0.025f, 0.035f};
	std::vector<float> s_down = {0.03f, 0.04f};
	std::vector<uint8_t> zp_gate = {127, 128};
	std::vector<uint8_t> zp_up = {126, 127};
	std::vector<uint8_t> zp_down = {125, 126};

	set_values(src_mem, to_f16(src));
	set_values(gate_mem, gate_q);
	set_values(up_mem, up_q);
	set_values(down_mem, down_q);
	set_values(s_gate_mem, to_f16(s_gate));
	set_values(s_up_mem, to_f16(s_up));
	set_values(s_down_mem, to_f16(s_down));
	set_values(zp_gate_mem, zp_gate);
	set_values(zp_up_mem, zp_up);
	set_values(zp_down_mem, zp_down);

	auto gate = dequantize_weights<uint8_t, uint8_t>(gate_q, ifm, hidden, s_gate, 2, 1, zp_gate, 2, 1);
	auto up = dequantize_weights<uint8_t, uint8_t>(up_q, ifm, hidden, s_up, 2, 1, zp_up, 2, 1);
	auto down = dequantize_weights<uint8_t, uint8_t>(down_q, hidden, ifm, s_down, 2, 1, zp_down, 2, 1);
	auto ref = gated_mlp_reference(src, gate, up, down, batch, ifm, hidden);

	auto net = create_compressed_network(engine, batch, ifm, hidden,
                                         src_mem, gate_mem, up_mem, down_mem, s_gate_mem, s_up_mem, s_down_mem, zp_gate_mem, zp_up_mem, zp_down_mem);
	auto outputs = net->execute();

	ASSERT_EQ(outputs.size(), size_t(1));
	ASSERT_EQ(outputs.begin()->first, "gmlp");
	check_output(outputs.at("gmlp").get_memory(), ref, 5e-2f);
}

TEST(gated_mlp_gpu, compressed_u8_per_oc_zp) {
	auto& engine = get_test_engine();
	skip_if_no_immad(engine);
	const int batch = 1;
	const int ifm = 4;
	const int hidden = 3;

	auto src_mem = engine.allocate_memory({{batch, 1, 1, ifm}, data_types::f16, format::bfyx});
	auto gate_mem = engine.allocate_memory({{ifm, hidden}, data_types::u8, format::bfyx});
	auto up_mem = engine.allocate_memory({{ifm, hidden}, data_types::u8, format::bfyx});
	auto down_mem = engine.allocate_memory({{hidden, ifm}, data_types::u8, format::bfyx});

	auto s_gate_mem = engine.allocate_memory({{1, hidden}, data_types::f16, format::bfyx});
	auto s_up_mem = engine.allocate_memory({{1, hidden}, data_types::f16, format::bfyx});
	auto s_down_mem = engine.allocate_memory({{1, ifm}, data_types::f16, format::bfyx});

	auto zp_gate_mem = engine.allocate_memory({{1, hidden}, data_types::u8, format::bfyx});
	auto zp_up_mem = engine.allocate_memory({{1, hidden}, data_types::u8, format::bfyx});
	auto zp_down_mem = engine.allocate_memory({{1, ifm}, data_types::u8, format::bfyx});

	std::vector<float> src = {0.6f, -0.4f, 0.2f, 1.0f};
	std::vector<uint8_t> gate_q = {121, 129, 132, 130, 131, 126, 127, 124, 128, 129, 126, 130};
	std::vector<uint8_t> up_q = {125, 128, 129, 130, 127, 126, 124, 131, 132, 128, 129, 127};
	std::vector<uint8_t> down_q = {130, 131, 132, 133, 129, 128, 127, 126, 125, 124, 123, 122};

	std::vector<float> s_gate = {0.05f, 0.10f, 0.15f};
	std::vector<float> s_up = {0.12f, 0.08f, 0.04f};
	std::vector<float> s_down = {0.03f, 0.06f, 0.09f, 0.12f};
	std::vector<uint8_t> zp_gate = {128, 127, 126};
	std::vector<uint8_t> zp_up = {127, 128, 129};
	std::vector<uint8_t> zp_down = {126, 127, 128, 129};

	set_values(src_mem, to_f16(src));
	set_values(gate_mem, gate_q);
	set_values(up_mem, up_q);
	set_values(down_mem, down_q);
	set_values(s_gate_mem, to_f16(s_gate));
	set_values(s_up_mem, to_f16(s_up));
	set_values(s_down_mem, to_f16(s_down));
	set_values(zp_gate_mem, zp_gate);
	set_values(zp_up_mem, zp_up);
	set_values(zp_down_mem, zp_down);

	auto gate = dequantize_weights<uint8_t, uint8_t>(gate_q, ifm, hidden, s_gate, 1, hidden, zp_gate, 1, hidden);
	auto up = dequantize_weights<uint8_t, uint8_t>(up_q, ifm, hidden, s_up, 1, hidden, zp_up, 1, hidden);
	auto down = dequantize_weights<uint8_t, uint8_t>(down_q, hidden, ifm, s_down, 1, ifm, zp_down, 1, ifm);
	auto ref = gated_mlp_reference(src, gate, up, down, batch, ifm, hidden);

	auto net = create_compressed_network(engine, batch, ifm, hidden,
                                         src_mem, gate_mem, up_mem, down_mem, s_gate_mem, s_up_mem, s_down_mem, zp_gate_mem, zp_up_mem, zp_down_mem);
	auto outputs = net->execute();

	ASSERT_EQ(outputs.size(), size_t(1));
	ASSERT_EQ(outputs.begin()->first, "gmlp");
	check_output(outputs.at("gmlp").get_memory(), ref, 3e-2f);
}

TEST(gated_mlp_gpu, compressed_i8_per_tensor_zp) {
	auto& engine = get_test_engine();
	skip_if_no_immad(engine);
	const int batch = 2;
	const int ifm = 4;
	const int hidden = 4;

	auto src_mem = engine.allocate_memory({{batch, 1, 1, ifm}, data_types::f16, format::bfyx});
	auto gate_mem = engine.allocate_memory({{ifm, hidden}, data_types::i8, format::bfyx});
	auto up_mem = engine.allocate_memory({{ifm, hidden}, data_types::i8, format::bfyx});
	auto down_mem = engine.allocate_memory({{hidden, ifm}, data_types::i8, format::bfyx});

	auto s_gate_mem = engine.allocate_memory({{1, 1}, data_types::f16, format::bfyx});
	auto s_up_mem = engine.allocate_memory({{1, 1}, data_types::f16, format::bfyx});
	auto s_down_mem = engine.allocate_memory({{1, 1}, data_types::f16, format::bfyx});

	std::vector<float> src = {0.1f, 0.2f, -0.3f, 0.4f,
							  -0.6f, 0.5f, 0.7f, -0.8f};
	std::vector<int8_t> gate_q = {2, 5, -3, 1,
								  4, -2, 6, 3,
								  -1, 2, 0, 5,
								  3, -4, 1, 2};
	std::vector<int8_t> up_q = {1, -2, 3, 4,
								5, 0, -1, 2,
								-3, 4, 1, -2,
								2, 3, -4, 0};
	std::vector<int8_t> down_q = {3, 2, 1, 0,
								  -1, -2, 4, 5,
								  2, -3, 1, -4,
								  0, 1, 2, 3};

	std::vector<float> s_gate = {0.2f};
	std::vector<float> s_up = {0.3f};
	std::vector<float> s_down = {0.25f};

	auto zp_gate_mem = engine.allocate_memory({{1, 1}, data_types::i8, format::bfyx});
	auto zp_up_mem = engine.allocate_memory({{1, 1}, data_types::i8, format::bfyx});
	auto zp_down_mem = engine.allocate_memory({{1, 1}, data_types::i8, format::bfyx});
	std::vector<int8_t> zp_gate = {0};
	std::vector<int8_t> zp_up = {1};
	std::vector<int8_t> zp_down = {-1};

	set_values(src_mem, to_f16(src));
	set_values(gate_mem, gate_q);
	set_values(up_mem, up_q);
	set_values(down_mem, down_q);
	set_values(s_gate_mem, to_f16(s_gate));
	set_values(s_up_mem, to_f16(s_up));
	set_values(s_down_mem, to_f16(s_down));
	set_values(zp_gate_mem, zp_gate);
	set_values(zp_up_mem, zp_up);
	set_values(zp_down_mem, zp_down);

	auto gate = dequantize_weights<int8_t, int8_t>(gate_q, ifm, hidden, s_gate, 1, 1, zp_gate, 1, 1);
	auto up = dequantize_weights<int8_t, int8_t>(up_q, ifm, hidden, s_up, 1, 1, zp_up, 1, 1);
	auto down = dequantize_weights<int8_t, int8_t>(down_q, hidden, ifm, s_down, 1, 1, zp_down, 1, 1);
	auto ref = gated_mlp_reference(src, gate, up, down, batch, ifm, hidden);

	auto net = create_compressed_network(engine, batch, ifm, hidden,
                                         src_mem, gate_mem, up_mem, down_mem, s_gate_mem, s_up_mem, s_down_mem, zp_gate_mem, zp_up_mem, zp_down_mem);
	auto outputs = net->execute();

	ASSERT_EQ(outputs.size(), size_t(1));
	ASSERT_EQ(outputs.begin()->first, "gmlp");
	check_output(outputs.at("gmlp").get_memory(), ref, 5e-2f);
}

TEST(gated_mlp_gpu, compressed_u4_per_tensor_zp_zero_scale) {
	auto& engine = get_test_engine();
	skip_if_no_immad(engine);
	const int batch = 2;
	const int ifm = 32;
	const int hidden = 32;

	auto src_mem = engine.allocate_memory({{batch, 1, 1, ifm}, data_types::f16, format::bfyx});
	auto gate_mem = engine.allocate_memory({{ifm, hidden}, data_types::u4, format::bfyx});
	auto up_mem = engine.allocate_memory({{ifm, hidden}, data_types::u4, format::bfyx});
	auto down_mem = engine.allocate_memory({{hidden, ifm}, data_types::u4, format::bfyx});

	auto s_gate_mem = engine.allocate_memory({{1, 1}, data_types::f16, format::bfyx});
	auto s_up_mem = engine.allocate_memory({{1, 1}, data_types::f16, format::bfyx});
	auto s_down_mem = engine.allocate_memory({{1, 1}, data_types::f16, format::bfyx});

	auto zp_gate_mem = engine.allocate_memory({{1, 1}, data_types::u8, format::bfyx});
	auto zp_up_mem = engine.allocate_memory({{1, 1}, data_types::u8, format::bfyx});
	auto zp_down_mem = engine.allocate_memory({{1, 1}, data_types::u8, format::bfyx});

	std::vector<float> src(batch * ifm, 0.0f);
	for (size_t i = 0; i < src.size(); ++i) {
		src[i] = (i % 3 == 0) ? -0.25f : 0.5f;
	}

	std::vector<uint8_t> gate_q(ifm * hidden / 2, 0x77);
	std::vector<uint8_t> up_q(ifm * hidden / 2, 0x66);
	std::vector<uint8_t> down_q(hidden * ifm / 2, 0x55);

	std::vector<float> s_gate = {0.0f};
	std::vector<float> s_up = {0.0f};
	std::vector<float> s_down = {0.0f};
	std::vector<uint8_t> zp_gate = {7};
	std::vector<uint8_t> zp_up = {6};
	std::vector<uint8_t> zp_down = {5};

	set_values(src_mem, to_f16(src));
	set_values(gate_mem, gate_q);
	set_values(up_mem, up_q);
	set_values(down_mem, down_q);
	set_values(s_gate_mem, to_f16(s_gate));
	set_values(s_up_mem, to_f16(s_up));
	set_values(s_down_mem, to_f16(s_down));
	set_values(zp_gate_mem, zp_gate);
	set_values(zp_up_mem, zp_up);
	set_values(zp_down_mem, zp_down);

	std::vector<float> ref(batch * ifm, 0.0f);
	auto net = create_compressed_network(engine, batch, ifm, hidden,
                                         src_mem, gate_mem, up_mem, down_mem, s_gate_mem, s_up_mem, s_down_mem, zp_gate_mem, zp_up_mem, zp_down_mem);
	auto outputs = net->execute();

	ASSERT_EQ(outputs.size(), size_t(1));
	ASSERT_EQ(outputs.begin()->first, "gmlp");
	check_output(outputs.at("gmlp").get_memory(), ref, 1e-3f);
}

TEST(gated_mlp_gpu, compressed_i4_per_tensor_zp_zero_scale) {
	auto& engine = get_test_engine();
	skip_if_no_immad(engine);
	const int batch = 2;
	const int ifm = 32;
	const int hidden = 32;

	auto src_mem = engine.allocate_memory({{batch, 1, 1, ifm}, data_types::f16, format::bfyx});
	auto gate_mem = engine.allocate_memory({{ifm, hidden}, data_types::i4, format::bfyx});
	auto up_mem = engine.allocate_memory({{ifm, hidden}, data_types::i4, format::bfyx});
	auto down_mem = engine.allocate_memory({{hidden, ifm}, data_types::i4, format::bfyx});

	auto s_gate_mem = engine.allocate_memory({{1, 1}, data_types::f16, format::bfyx});
	auto s_up_mem = engine.allocate_memory({{1, 1}, data_types::f16, format::bfyx});
	auto s_down_mem = engine.allocate_memory({{1, 1}, data_types::f16, format::bfyx});

	auto zp_gate_mem = engine.allocate_memory({{1, 1}, data_types::i8, format::bfyx});
	auto zp_up_mem = engine.allocate_memory({{1, 1}, data_types::i8, format::bfyx});
	auto zp_down_mem = engine.allocate_memory({{1, 1}, data_types::i8, format::bfyx});

	std::vector<float> src(batch * ifm, 0.0f);
	for (size_t i = 0; i < src.size(); ++i) {
		src[i] = (i % 4 == 0) ? 0.125f : -0.375f;
	}

	std::vector<uint8_t> gate_q(ifm * hidden / 2, 0x11);
	std::vector<uint8_t> up_q(ifm * hidden / 2, 0x22);
	std::vector<uint8_t> down_q(hidden * ifm / 2, 0x33);

	std::vector<float> s_gate = {0.0f};
	std::vector<float> s_up = {0.0f};
	std::vector<float> s_down = {0.0f};
	std::vector<int8_t> zp_gate = {1};
	std::vector<int8_t> zp_up = {2};
	std::vector<int8_t> zp_down = {3};

	set_values(src_mem, to_f16(src));
	set_values(gate_mem, gate_q);
	set_values(up_mem, up_q);
	set_values(down_mem, down_q);
	set_values(s_gate_mem, to_f16(s_gate));
	set_values(s_up_mem, to_f16(s_up));
	set_values(s_down_mem, to_f16(s_down));
	set_values(zp_gate_mem, zp_gate);
	set_values(zp_up_mem, zp_up);
	set_values(zp_down_mem, zp_down);

	std::vector<float> ref(batch * ifm, 0.0f);
	auto net = create_compressed_network(engine, batch, ifm, hidden,
                                         src_mem, gate_mem, up_mem, down_mem, s_gate_mem, s_up_mem, s_down_mem, zp_gate_mem, zp_up_mem, zp_down_mem);
	auto outputs = net->execute();

	ASSERT_EQ(outputs.size(), size_t(1));
	ASSERT_EQ(outputs.begin()->first, "gmlp");
	check_output(outputs.at("gmlp").get_memory(), ref, 1e-3f);
}

TEST(gated_mlp_gpu, compressed_u4_grouped_zp_zero_scale) {
	auto& engine = get_test_engine();
	skip_if_no_immad(engine);
	const int batch = 8;
	const int ifm = 48;
	const int hidden = 48;

	auto src_mem = engine.allocate_memory({{batch, 1, 1, ifm}, data_types::f16, format::bfyx});
	auto gate_mem = engine.allocate_memory({{ifm, hidden}, data_types::u4, format::bfyx});
	auto up_mem = engine.allocate_memory({{ifm, hidden}, data_types::u4, format::bfyx});
	auto down_mem = engine.allocate_memory({{hidden, ifm}, data_types::u4, format::bfyx});

	auto s_gate_mem = engine.allocate_memory({{2, 1}, data_types::f16, format::bfyx});
	auto s_up_mem = engine.allocate_memory({{2, 1}, data_types::f16, format::bfyx});
	auto s_down_mem = engine.allocate_memory({{2, 1}, data_types::f16, format::bfyx});

	auto zp_gate_mem = engine.allocate_memory({{2, 1}, data_types::u8, format::bfyx});
	auto zp_up_mem = engine.allocate_memory({{2, 1}, data_types::u8, format::bfyx});
	auto zp_down_mem = engine.allocate_memory({{2, 1}, data_types::u8, format::bfyx});

	std::vector<float> src(batch * ifm, 0.0f);
	for (size_t i = 0; i < src.size(); ++i) {
		src[i] = (i % 6 == 0) ? -0.2f : 0.4f;
	}

	std::vector<uint8_t> gate_q(ifm * hidden / 2, 0x78);
	std::vector<uint8_t> up_q(ifm * hidden / 2, 0x67);
	std::vector<uint8_t> down_q(hidden * ifm / 2, 0x56);

	std::vector<float> s_gate = {0.02f, 0.03f};
	std::vector<float> s_up = {0.03f, 0.04f};
	std::vector<float> s_down = {0.04f, 0.05f};
	std::vector<uint8_t> zp_gate = {7, 8};
	std::vector<uint8_t> zp_up = {6, 7};
	std::vector<uint8_t> zp_down = {5, 6};

	set_values(src_mem, to_f16(src));
	set_values(gate_mem, gate_q);
	set_values(up_mem, up_q);
	set_values(down_mem, down_q);
	set_values(s_gate_mem, to_f16(s_gate));
	set_values(s_up_mem, to_f16(s_up));
	set_values(s_down_mem, to_f16(s_down));
	set_values(zp_gate_mem, zp_gate);
	set_values(zp_up_mem, zp_up);
	set_values(zp_down_mem, zp_down);

	auto net = create_compressed_network(engine, batch, ifm, hidden,
                                         src_mem, gate_mem, up_mem, down_mem, s_gate_mem, s_up_mem, s_down_mem, zp_gate_mem, zp_up_mem, zp_down_mem, "gmlp_u4_grouped");
	auto outputs = net->execute();

	ASSERT_EQ(outputs.size(), size_t(1));
	ASSERT_EQ(outputs.begin()->first, "gmlp_u4_grouped");
	check_output_finite(outputs.at("gmlp_u4_grouped").get_memory());
}

TEST(gated_mlp_gpu, compressed_i4_grouped_zp_zero_scale) {
	auto& engine = get_test_engine();
	skip_if_no_immad(engine);
	const int batch = 8;
	const int ifm = 48;
	const int hidden = 48;

	auto src_mem = engine.allocate_memory({{batch, 1, 1, ifm}, data_types::f16, format::bfyx});
	auto gate_mem = engine.allocate_memory({{ifm, hidden}, data_types::i4, format::bfyx});
	auto up_mem = engine.allocate_memory({{ifm, hidden}, data_types::i4, format::bfyx});
	auto down_mem = engine.allocate_memory({{hidden, ifm}, data_types::i4, format::bfyx});

	auto s_gate_mem = engine.allocate_memory({{2, 1}, data_types::f16, format::bfyx});
	auto s_up_mem = engine.allocate_memory({{2, 1}, data_types::f16, format::bfyx});
	auto s_down_mem = engine.allocate_memory({{2, 1}, data_types::f16, format::bfyx});

	auto zp_gate_mem = engine.allocate_memory({{2, 1}, data_types::i8, format::bfyx});
	auto zp_up_mem = engine.allocate_memory({{2, 1}, data_types::i8, format::bfyx});
	auto zp_down_mem = engine.allocate_memory({{2, 1}, data_types::i8, format::bfyx});

	std::vector<float> src(batch * ifm, 0.0f);
	for (size_t i = 0; i < src.size(); ++i) {
		src[i] = (i % 7 == 0) ? 0.35f : -0.15f;
	}

	std::vector<uint8_t> gate_q(ifm * hidden / 2, 0x12);
	std::vector<uint8_t> up_q(ifm * hidden / 2, 0x23);
	std::vector<uint8_t> down_q(hidden * ifm / 2, 0x34);

	std::vector<float> s_gate = {0.02f, 0.03f};
	std::vector<float> s_up = {0.03f, 0.04f};
	std::vector<float> s_down = {0.04f, 0.05f};
	std::vector<int8_t> zp_gate = {1, 2};
	std::vector<int8_t> zp_up = {2, 3};
	std::vector<int8_t> zp_down = {3, 4};

	set_values(src_mem, to_f16(src));
	set_values(gate_mem, gate_q);
	set_values(up_mem, up_q);
	set_values(down_mem, down_q);
	set_values(s_gate_mem, to_f16(s_gate));
	set_values(s_up_mem, to_f16(s_up));
	set_values(s_down_mem, to_f16(s_down));
	set_values(zp_gate_mem, zp_gate);
	set_values(zp_up_mem, zp_up);
	set_values(zp_down_mem, zp_down);

	auto net = create_compressed_network(engine, batch, ifm, hidden,
                                         src_mem, gate_mem, up_mem, down_mem, s_gate_mem, s_up_mem, s_down_mem, zp_gate_mem, zp_up_mem, zp_down_mem, "gmlp_i4_grouped");
	auto outputs = net->execute();

	ASSERT_EQ(outputs.size(), size_t(1));
	ASSERT_EQ(outputs.begin()->first, "gmlp_i4_grouped");
	check_output_finite(outputs.at("gmlp_i4_grouped").get_memory());
}
#endif
