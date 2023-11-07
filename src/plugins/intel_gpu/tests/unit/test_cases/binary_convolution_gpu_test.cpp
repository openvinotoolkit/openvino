// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/binary_convolution.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include <intel_gpu/primitives/data.hpp>

#include <iostream>

using namespace cldnn;
using namespace ::tests;

// Batch, groups, IC, IW, IH, OC, OW, OH, KH, KW, SH, SW, PH, PW
struct TestParams {
    int b;
    int g;

    int ic;
    int ih;
    int iw;

    int oc;
    int oh;
    int ow;

    int kh;
    int kw;

    int sh;
    int sw;

    int ph;
    int pw;

    float pad_value;
    data_types dt;
    std::string name;
    bool is_caching_test;

    bool isConsistent() const
    {
        bool res = true;

        res &= (((iw - kw + 2*pw) / sw + 1) == ow);
        res &= (((ih - kh + 2*ph) / sh + 1) == oh);
        return res;
    }

    friend ::std::ostream& operator<<(::std::ostream& os, const TestParams& p) {
        return os << "Params: [ b=" << p.b
                  << "; g=" << p.g
                  << "; src=[" << p.ic << "; " << p.ih << "; " << p.iw << "]"
                  << "; dst=[" << p.oc << "; " << p.oh << "; " << p.ow << "]"
                  << "; k=[" << p.kh << "; " << p.kw << "]"
                  << "; stride=[" << p.sh << "; " << p.sw << "]"
                  << "; pad=[" << p.ph << "; " << p.pw << "]"
                  << "; pad_value=" << p.pad_value
                  << "; name=" << p.name
                  << "; is_caching_test=" << p.is_caching_test
                  << "]";
    }
    friend void PrintTo(const TestParams& p, ::std::ostream* os) {
        *os << p;
    }
};

static void fill(cldnn::memory::ptr mem) {
    cldnn::mem_lock<uint32_t> ptr(mem, get_test_stream());
    for (size_t i = 0; i < div_up(mem->get_layout().count(), 32); i++) {
        ptr[i] = (uint32_t)rand() % (1 << 31);
    }
}

template <typename data_t_src, typename data_t_wei,
          typename data_t_acc, typename data_t_dst>
void compute_ref_conv_bin(const cldnn::memory::ptr src,
                          const cldnn::memory::ptr weights,
                          cldnn::memory::ptr dst,
                          TestParams &p) {

    cldnn::mem_lock<data_t_src> src_data(src, get_test_stream());
    cldnn::mem_lock<data_t_wei> weights_data(weights, get_test_stream());
    cldnn::mem_lock<data_t_dst> dst_data(dst, get_test_stream());

    int pack_size = sizeof(data_t_src) * 8;

    int B = p.b;
    int NG = p.g;
    int IC = p.ic;
    int IH = p.ih;
    int IW = p.iw;

    int OC = p.oc;
    int OH = p.oh;
    int OW = p.ow;

    int KH = p.kh;
    int KW = p.kw;

    int SH = p.sh;
    int SW = p.sw;

    int PH = p.ph;
    int PW = p.pw;

    auto extract_bit = [&](data_t_src val, data_t_src bit) -> data_t_src {
        return (data_t_src)((val >> bit) & 0x1);
    };

    auto ker = [&](data_t_acc &d, int g, int mb, int oc,int oh, int ow, int& ks) {
        for (int ic = 0; ic < IC / NG; ++ic) {
            for (int kh = 0; kh < KH; ++kh)
                for (int kw = 0; kw < KW; ++kw) {
                    const int ih = oh * SH - PH + kh;
                    const int iw = ow * SW - PW + kw;

                    int widx =   g * OC / NG *IC / NG * KH * KW
                                 + oc * IC / NG * KH * KW
                                 + ic * KH * KW
                                 + kh * KW
                                 + kw;
                    int iidx = -1;
                    uint8_t w = extract_bit(weights_data[widx / pack_size], widx % pack_size);
                    uint8_t s = 0;

                    if ((ih < 0 || ih >= IH || iw < 0 || iw >= IW)) {
                        if (p.pad_value == 0.0f)
                            continue;
                        else
                            s = (p.pad_value == -1.0f) ? 0 : 1;
                    } else {
                        if (ic == 0) ks++;
                        iidx = mb * div_up(IC, pack_size) * IH * IW
                               + g * div_up(IC, pack_size) / NG * IH * IW
                               + (ic/pack_size) * IH * IW
                               + ih * IW
                               + iw;

                        s = extract_bit(src_data[iidx], ic % pack_size);
                    }
                    d += (data_t_acc)(s ^ w);
                }
        }
    };

    for (int g = 0; g < NG; g++) {
        for (int b = 0; b < B; b++) {
            for (int oc = 0; oc < OC / NG; oc++) {
                for (int oh = 0; oh < OH; oh++) {
                    for (int ow = 0; ow < OW; ow++) {
                        data_t_acc a = 0;
                        int ks = 0;
                        ker(a, g, b, oc, oh, ow, ks);
                        int dst_off = b * OC * OH* OW
                                      + g * OC / NG * OH * OW
                                      + oc * OH * OW
                                      + oh * OW
                                      + ow;
                        if (p.pad_value == 0.0f)
                            dst_data[dst_off] =(data_t_dst)(IC*ks - 2*a);
                        else
                            dst_data[dst_off] = (data_t_dst)(IC*KH*KW - 2*a);
                    }
                }
            }
        }
    }
}

class binary_convolution_test : public ::testing::TestWithParam<TestParams> {
    void SetUp() override {
        std::cout << GetParam() << std::endl;
        ASSERT_TRUE(GetParam().isConsistent());
    }
};

TEST_P(binary_convolution_test, conv) {
    auto& engine = get_test_engine();

    // DG2 is not validated for binary convolution: https://github.com/openvinotoolkit/openvino/pull/12486
    if(engine.get_device_info().supports_immad)
        return;

    ov::intel_gpu::ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    topology topology_bin;

    std::string weights_suffix = "_w_";

    std::string input_name = "input";
    std::string output_name = "conv";

    TestParams p = GetParam();

    ov::Strides stride = {static_cast<uint64_t>(p.sh), static_cast<uint64_t>(p.sw)};
    ov::CoordinateDiff pad = {p.ph, p.pw};
    ov::Strides dilation = {1,1};

    cldnn::tensor is_size{ cldnn::batch(p.b),
                           cldnn::feature(p.ic),
                           cldnn::spatial(p.iw, p.ih) };
    cldnn::tensor wei_size{ cldnn::batch(p.oc),
                            cldnn::feature(p.ic),
                            cldnn::spatial(p.kw, p.kh) };
    cldnn::tensor os_size{ cldnn::batch(p.b),
                            cldnn::feature(p.oc),
                            cldnn::spatial(p.ow, p.oh)};

    auto input       = engine.allocate_memory({ cldnn::data_types::u1, cldnn::format::b_fs_yx_32fp, is_size });
    auto weights     = engine.allocate_memory({ cldnn::data_types::u1, cldnn::format::bfyx, wei_size });
    auto output_ref  = engine.allocate_memory({ cldnn::data_types::f32, cldnn::format::bfyx, os_size });

    fill(input);
    fill(weights);

    compute_ref_conv_bin<uint32_t, uint32_t, int32_t, float>(input, weights, output_ref, p);

//    print_bin_blob(input,"input");
//    print_bin_blob_packed(input,"input");
//    print_bin_blob(weights, "weights");
//    print_blob(output_ref, "ref_out");

    topology_bin.add(input_layout(input_name, input->get_layout()));
    topology_bin.add(data(output_name + weights_suffix, weights));

    topology_bin.add(binary_convolution(output_name, input_info(input_name), {output_name + weights_suffix},
                                        stride, pad, dilation, os_size, 1, p.pad_value, p.dt));

    cldnn::network::ptr network_bin = get_network(engine, topology_bin, config, get_test_stream_ptr(), p.is_caching_test);

    network_bin->set_input_data(input_name, input);

    std::map<primitive_id, network_output> outputs = network_bin->execute();
    auto outputMemory = outputs.at(output_name).get_memory();

    for (size_t i = 0; i < output_ref->count(); i++) {
        if (p.dt == data_types::f32) {
            cldnn::mem_lock<float> ref(output_ref, get_test_stream());
            cldnn::mem_lock<float> opt(outputMemory, get_test_stream());

            ASSERT_EQ(ref[i], opt[i]) << i;
        } else if (p.dt == data_types::f16) {
            cldnn::mem_lock<float> ref(output_ref, get_test_stream());
            cldnn::mem_lock<uint16_t> opt(outputMemory, get_test_stream());

            ASSERT_EQ(ref[i], half_to_float(opt[i])) << i;
        }
    }
}

// Batch, groups, IC, IW, IH, OC, OW, OH, KH, KW, SH, SW, PH, PW
INSTANTIATE_TEST_SUITE_P(BinaryConvTest, binary_convolution_test, ::testing::Values(
        TestParams{1, 1,  16,2,2,   4,2,2, 3,3, 1,1, 1,1, -1.0f, data_types::f32, "small", false},
        TestParams{1, 1,  17,2,2,   4,2,2, 3,3, 1,1, 1,1, -1.0f, data_types::f32, "small", false},
        TestParams{1, 1,  17,2,2,   4,2,2, 3,3, 1,1, 1,1,  0.0f, data_types::f32, "small", false},
        TestParams{1, 1,  17,2,2,   4,2,2, 3,3, 1,1, 1,1,  1.0f, data_types::f32, "small", false},
        TestParams{1, 1,  16,2,2,  16,2,2, 3,3, 1,1, 1,1,  1.0f, data_types::f32, "small", false},
        TestParams{1, 1,  32,2,2,  32,2,2, 3,3, 1,1, 1,1,  1.0f, data_types::f32, "small", false},
        TestParams{1, 1,  32,2,2,  32,2,2, 1,1, 1,1, 0,0,  1.0f, data_types::f32, "small", false},
        TestParams{1, 1, 128,2,2, 128,2,2, 1,1, 1,1, 0,0, -1.0f, data_types::f32, "small", false},
        TestParams{1, 1,  16,4,3,   4,4,3, 1,1, 1,1, 0,0, -1.0f, data_types::f32, "small", false},
        TestParams{1, 1,  16,2,2,   4,2,2, 3,3, 1,1, 1,1, -1.0f, data_types::f16, "small", false},
        TestParams{1, 1,  17,2,2,   4,2,2, 3,3, 1,1, 1,1, -1.0f, data_types::f16, "small", false},
        TestParams{1, 1,  17,2,2,   4,2,2, 3,3, 1,1, 1,1,  0.0f, data_types::f16, "small", false},
        TestParams{1, 1,  17,2,2,   4,2,2, 3,3, 1,1, 1,1,  1.0f, data_types::f16, "small", false},
        TestParams{1, 1,  16,2,2,  16,2,2, 3,3, 1,1, 1,1,  1.0f, data_types::f16, "small", false},
        TestParams{1, 1,  32,2,2,  32,2,2, 3,3, 1,1, 1,1,  1.0f, data_types::f16, "small", false},
        TestParams{1, 1,  32,2,2,  32,2,2, 1,1, 1,1, 0,0,  1.0f, data_types::f16, "small", false},
        TestParams{1, 1, 128,2,2, 128,2,2, 1,1, 1,1, 0,0, -1.0f, data_types::f16, "small", false},
        TestParams{1, 1,  16,4,3,   4,4,3, 1,1, 1,1, 0,0, -1.0f, data_types::f16, "small", false},
        TestParams{1, 1,  9,16,32, 17,8,16, 7,7, 2,2, 3,3, -1.0f, data_types::f16, "small", false},
        TestParams{1, 1,  9,16,32, 17,8,16, 7,7, 2,2, 3,3, 1.0f, data_types::f16, "small", false},

        // Resnet-18 3x3
        TestParams{1, 1,  64,56,56,  64,56,56, 3,3, 1,1, 1,1, -1.0f, data_types::f16, "resnet18_0", false},
        TestParams{1, 1,  64,56,56, 128,28,28, 3,3, 2,2, 1,1, -1.0f, data_types::f16, "resnet18_1", false},
        TestParams{1, 1, 128,28,28, 128,28,28, 3,3, 1,1, 1,1, -1.0f, data_types::f16, "resnet18_2", false},
        TestParams{1, 1, 128,28,28, 256,14,14, 3,3, 2,2, 1,1, -1.0f, data_types::f16, "resnet18_3", false},
        TestParams{1, 1, 256,14,14, 256,14,14, 3,3, 1,1, 1,1, -1.0f, data_types::f16, "resnet18_4", false},
        TestParams{1, 1, 256,14,14, 512, 7, 7, 3,3, 2,2, 1,1, -1.0f, data_types::f16, "resnet18_5", false},
        TestParams{1, 1, 512, 7, 7, 512, 7, 7, 3,3, 1,1, 1,1, -1.0f, data_types::f16, "resnet18_6", false},
        // Resnet-50
        TestParams{1, 1, 64,56,56, 64,56,56, 1,1, 1,1, 0,0, -1.0f, data_types::f16, "resnet50_0", false},
        TestParams{1, 1, 64,56,56, 256,56,56, 1,1, 1,1, 0,0, -1.0f, data_types::f16, "resnet50_1", false},
        TestParams{1, 1, 256,56,56, 128,28,28, 1,1, 2,2, 0,0, -1.0f, data_types::f16, "resnet50_2", false},
        TestParams{1, 1, 128,28,28, 512,28,28, 1,1, 1,1, 0,0, -1.0f, data_types::f16, "resnet50_3", false},
        TestParams{1, 1, 512,28,28, 128,28,28, 1,1, 1,1, 0,0, -1.0f, data_types::f16, "resnet50_4", false},
        TestParams{1, 1, 512,28,28, 256,14,14, 1,1, 2,2, 0,0, -1.0f, data_types::f16, "resnet50_5", false},
        TestParams{1, 1, 256,14,14, 1024,14,14, 1,1, 1,1, 0,0, -1.0f, data_types::f16, "resnet50_6", false},
        TestParams{1, 1, 1024,14,14, 256,14,14, 1,1, 1,1, 0,0, -1.0f, data_types::f16, "resnet50_7", false},
        TestParams{1, 1, 1024,14,14, 512,7,7, 1,1, 2,2, 0,0, -1.0f, data_types::f16, "resnet50_8", false},
        TestParams{1, 1, 512,7,7, 2048,7,7, 1,1, 1,1, 0,0, -1.0f, data_types::f16, "resnet50_9", false},
        TestParams{1, 1, 2048,7,7, 512,7,7, 1,1, 1,1, 0,0, -1.0f, data_types::f16, "resnet50_10", false},
        // Mobilenet-ssd-vd
        TestParams{1, 1,  56,96,168, 112,96,168, 1,1, 1,1, 0,0, -1.0f, data_types::f16, "conv2_2_sep_BIN", false}, // back_bone_seq_conv2_2_sep_BIN
        TestParams{1, 1, 112,96,168, 112,96,168, 1,1, 1,1, 0,0, -1.0f, data_types::f16, "conv3_1_sep_BIN", false}, // back_bone_seq_conv3_1_sep_BIN
        TestParams{1, 1,  112,48,84, 208,48, 84, 1,1, 1,1, 0,0, -1.0f, data_types::f16, "conv3_2_sep_BIN", false}, // back_bone_seq_conv3_2_sep_BIN
        TestParams{1, 1,  208,48,84, 216,48, 84, 1,1, 1,1, 0,0, -1.0f, data_types::f16, "conv4_1_sep_BIN", false}, // back_bone_seq_conv4_1_sep_BIN
        TestParams{1, 1,  216,24,42, 328,24, 42, 1,1, 1,1, 0,0, -1.0f, data_types::f16, "conv4_2_sep_BIN", false}, // back_bone_seq_conv4_2_sep_BIN
        TestParams{1, 1,  328,24,42, 288,24, 42, 1,1, 1,1, 0,0, -1.0f, data_types::f16, "conv5_1_sep_BIN", false}, // back_bone_seq_conv5_1_sep_BIN
        TestParams{1, 1,  288,24,42, 288,24, 42, 1,1, 1,1, 0,0, -1.0f, data_types::f16, "conv5_2_sep_BIN", false}, // back_bone_seq_conv5_2_sep_BIN
        TestParams{1, 1,  288,24,42, 240,24, 42, 1,1, 1,1, 0,0, -1.0f, data_types::f16, "conv5_3_sep_BIN", false}, // back_bone_seq_conv5_3_sep_BIN
        TestParams{1, 1,  240,24,42, 264,24, 42, 1,1, 1,1, 0,0, -1.0f, data_types::f16, "conv5_4_sep_BIN", false}, // back_bone_seq_conv5_4_sep_BIN
        TestParams{1, 1,  264,24,42, 192,24, 42, 1,1, 1,1, 0,0, -1.0f, data_types::f16, "conv5_5_sep_BIN", false}, // back_bone_seq_conv5_5_sep_BIN
        TestParams{1, 1,  192,12,21, 208,12, 21, 1,1, 1,1, 0,0, -1.0f, data_types::f16, "conv5_6_sep_BIN", false}, // back_bone_seq_conv5_6_sep_BIN
        TestParams{1, 1,  208,12,21,  88,12, 21, 1,1, 1,1, 0,0, -1.0f, data_types::f16, "conv6_sep_BN", false} // back_bone_seq_conv6_sep_BN
));

INSTANTIATE_TEST_SUITE_P(export_import, binary_convolution_test, ::testing::Values(
        TestParams{1, 1,  208,12,21,  88,12, 21, 1,1, 1,1, 0,0, -1.0f, data_types::f16, "conv6_sep_BN", true}
));

template <typename T>
static void set_binary_values(cldnn::memory::ptr mem, std::vector<T> args) {
    cldnn::mem_lock<T> ptr(mem, get_test_stream());

    auto it = ptr.begin();
    for (auto x : args)
        *it++ = x;
}

TEST(binary_convolution, basic_convolution_1x1_single_packed_channel) {
    auto& engine = get_test_engine();
    // DG2 is not validated for binary convolution: https://github.com/openvinotoolkit/openvino/pull/12486
    if(engine.get_device_info().supports_immad)
        return;

    auto input = engine.allocate_memory({ data_types::u1, format::b_fs_yx_32fp, { 1, 16, 2, 2 } });
    auto weights = engine.allocate_memory({ data_types::u1, format::bfyx, { 4, 16, 1, 1 } });

    // 0 0 1 0  0 1 0 0  1 0 1 0  1 0 1 0
    // 1 0 0 0  0 1 1 0  0 1 1 0  1 0 1 0
    // 1 1 0 0  1 0 1 1  1 1 1 1  1 0 1 0
    // 0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 1
    set_binary_values<uint32_t>(input, { 21796, 22113, 24531, 32768 });

    // 1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1
    // 0 1 0 1  0 1 0 1  1 0 1 0  1 0 1 0
    // 1 0 1 0  1 0 1 0  0 1 0 1  0 1 0 1
    // 0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
    set_binary_values<uint16_t>(weights, { 65535, 21930, 43605, 0 });

    // 16 - 2*popcount(1 1 0 1  1 0 1 1  0 1 0 1  0 1 0 1) = -4
    // 16 - 2*popcount(0 1 1 1  1 0 0 1  1 0 0 1  0 1 0 1) = -2
    // 16 - 2*popcount(0 0 1 1  0 1 0 0  0 0 0 0  0 1 0 1) = 6
    // 16 - 2*popcount(1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 0) = -14

    // 16 - 2*popcount(0 1 1 1  0 0 0 1  0 0 0 0  0 0 0 0) = 8
    // 16 - 2*popcount(1 1 0 1  0 0 1 1  1 1 0 0  0 0 0 0) = 2
    // 16 - 2*popcount(1 0 0 1  1 1 1 0  0 1 0 1  0 0 0 0) = 2
    // 16 - 2*popcount(0 1 0 1  0 1 0 1  1 0 1 0  1 0 1 1) = -2

    // 16 - 2*popcount(1 0 0 0  1 1 1 0  1 1 1 1  1 1 1 1) = -8
    // 16 - 2*popcount(0 0 1 0  1 1 0 0  0 0 1 1  1 1 1 1) = -2
    // 16 - 2*popcount(0 1 1 0  0 0 0 1  1 0 1 0  1 1 1 1) = -2
    // 16 - 2*popcount(1 0 1 0  1 0 1 0  0 1 0 1  0 1 0 0) = 2

    // 16 - 2*popcount(0 0 1 0  0 1 0 0  1 0 1 0  1 0 1 0) = 4
    // 16 - 2*popcount(1 0 0 0  0 1 1 0  0 1 1 0  1 0 1 0) = 2
    // 16 - 2*popcount(1 1 0 0  1 0 1 1  1 1 1 1  1 0 1 0) = -6
    // 16 - 2*popcount(0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 1) = 14
    VF<float> output_vec = {
            -4.0f, -2.0f,  6.0f, -14.0f,
             8.0f,  2.0f,  2.0f,  -2.0f,
            -8.0f, -2.0f, -2.0f,   2.0f,
             4.0f,  2.0f, -6.0f,  14.0f };

    topology topology(
            input_layout("input", input->get_layout()),
            data("weights", weights),
            binary_convolution("binary_conv", input_info("input"), { "weights" },
                               { 1,1 },
                               { 0,0 },
                               { 1,1 },
                               { 1,4,2,2 },
                               0, 0.0f,
                               data_types::f32,
                               padding{ { 0,0,0,0 }, 0 })
    );

    ov::intel_gpu::ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));

    network network(engine, topology, config);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "binary_conv");

    auto output_memory = outputs.at("binary_conv").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());

    ASSERT_EQ(output_layout.format, format::bfyx);
    ASSERT_EQ(output_layout.data_type, data_types::f32);
    ASSERT_EQ(output_layout.batch(), 1);
    ASSERT_EQ(output_layout.feature(), 4);
    ASSERT_EQ(output_layout.spatial(1), 2);
    ASSERT_EQ(output_layout.spatial(0), 2);

    for (size_t i = 0; i < output_layout.count(); i++)
    {
        ASSERT_EQ(output_ptr[i], output_vec[i]) << "index="<< i;
    }
}

TEST(binary_convolution, basic_convolution_1x1_single_packed_channel_fp16) {
    auto& engine = get_test_engine();
    // DG2 is not validated for binary convolution: https://github.com/openvinotoolkit/openvino/pull/12486
    if(engine.get_device_info().supports_immad)
        return;

    auto input = engine.allocate_memory({ data_types::u1, format::b_fs_yx_32fp, { 1, 16, 2, 2 } });
    auto weights = engine.allocate_memory({ data_types::u1, format::bfyx, { 4, 16, 1, 1 } });

    // 0 0 1 0  0 1 0 0  1 0 1 0  1 0 1 0
    // 1 0 0 0  0 1 1 0  0 1 1 0  1 0 1 0
    // 1 1 0 0  1 0 1 1  1 1 1 1  1 0 1 0
    // 0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 1
    set_binary_values<uint32_t>(input, { 21796, 22113, 24531, 32768 });

    // 1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1
    // 0 1 0 1  0 1 0 1  1 0 1 0  1 0 1 0
    // 1 0 1 0  1 0 1 0  0 1 0 1  0 1 0 1
    // 0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
    set_binary_values<uint16_t>(weights, { 65535, 21930, 43605, 0 });

    // 16 - 2*popcount(1 1 0 1  1 0 1 1  0 1 0 1  0 1 0 1) = -4
    // 16 - 2*popcount(0 1 1 1  1 0 0 1  1 0 0 1  0 1 0 1) = -2
    // 16 - 2*popcount(0 0 1 1  0 1 0 0  0 0 0 0  0 1 0 1) = 6
    // 16 - 2*popcount(1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 0) = -14

    // 16 - 2*popcount(0 1 1 1  0 0 0 1  0 0 0 0  0 0 0 0) = 8
    // 16 - 2*popcount(1 1 0 1  0 0 1 1  1 1 0 0  0 0 0 0) = 2
    // 16 - 2*popcount(1 0 0 1  1 1 1 0  0 1 0 1  0 0 0 0) = 2
    // 16 - 2*popcount(0 1 0 1  0 1 0 1  1 0 1 0  1 0 1 1) = -2

    // 16 - 2*popcount(1 0 0 0  1 1 1 0  1 1 1 1  1 1 1 1) = -8
    // 16 - 2*popcount(0 0 1 0  1 1 0 0  0 0 1 1  1 1 1 1) = -2
    // 16 - 2*popcount(0 1 1 0  0 0 0 1  1 0 1 0  1 1 1 1) = -2
    // 16 - 2*popcount(1 0 1 0  1 0 1 0  0 1 0 1  0 1 0 0) = 2

    // 16 - 2*popcount(0 0 1 0  0 1 0 0  1 0 1 0  1 0 1 0) = 4
    // 16 - 2*popcount(1 0 0 0  0 1 1 0  0 1 1 0  1 0 1 0) = 2
    // 16 - 2*popcount(1 1 0 0  1 0 1 1  1 1 1 1  1 0 1 0) = -6
    // 16 - 2*popcount(0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 1) = 14
    VF<float> output_vec = {
            -4.0f, -2.0f,  6.0f, -14.0f,
             8.0f,  2.0f,  2.0f,  -2.0f,
            -8.0f, -2.0f, -2.0f,   2.0f,
             4.0f,  2.0f, -6.0f,  14.0f };

    topology topology(
            input_layout("input", input->get_layout()),
            data("weights", weights),
            binary_convolution("binary_conv", input_info("input"), { "weights" },
                               { 1,1 },
                               { 0,0 },
                               { 1,1 },
                               { 1,4,2,2 },
                               0, 0.0f,
                               data_types::f16,
                               padding{ { 0,0,0,0 }, 0 })
    );

    ov::intel_gpu::ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));

    network network(engine, topology, config);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "binary_conv");

    auto output_memory = outputs.at("binary_conv").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<uint16_t> output_ptr(output_memory, get_test_stream());

    ASSERT_EQ(output_layout.format, format::bfyx);
    ASSERT_EQ(output_layout.data_type, data_types::f16);
    ASSERT_EQ(output_layout.batch(), 1);
    ASSERT_EQ(output_layout.feature(), 4);
    ASSERT_EQ(output_layout.spatial(1), 2);
    ASSERT_EQ(output_layout.spatial(0), 2);

    for (size_t i = 0; i < output_layout.count(); i++) {
        ASSERT_EQ(half_to_float(output_ptr[i]), output_vec[i]) << "index="<< i;
    }
}
