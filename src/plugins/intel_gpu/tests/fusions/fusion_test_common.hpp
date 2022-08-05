// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/graph/network.hpp>

#include <cmath>

using namespace cldnn;
using namespace ::tests;

template<typename T>
class BaseFusingTest : public ::testing::TestWithParam<T> {
public:
#ifdef ENABLE_ONEDNN_FOR_GPU
    cldnn::engine& engine = get_onednn_test_engine();
#else
    cldnn::engine& engine = get_test_engine();
#endif
    cldnn::topology topology_fused;
    cldnn::topology topology_non_fused;
    cldnn::build_options bo_fused;
    cldnn::build_options bo_not_fused;

    float tolerance = 0.0f;

    static const int min_random = -200;
    static const int max_random = 200;

    void SetUp() override {
        bo_fused.set_option(build_option::optimize_data(true));
        bo_not_fused.set_option(build_option::optimize_data(false));
        bo_not_fused.set_option(build_option::allow_static_input_reorder(true));
    }

    void compare(network& not_fused, network& fused, T& p, bool count_reorder = false) {
        auto outputs_ref = not_fused.execute();
        auto outputs_fused = fused.execute();
        auto get_reorders_count = [](network& net) -> size_t {
            size_t count = 0;
            for (auto& pi : net.get_primitives_info()) {
                if (pi.type_id == "reorder") {
                    auto exec_prims = net.get_executed_primitives();
                    auto it = std::find_if(exec_prims.begin(), exec_prims.end(), [&](const std::pair<primitive_id, event::ptr>& e) -> bool {
                        return e.first == pi.original_id;
                    });
                    // We count executed reorders only
                    if (it != exec_prims.end())
                        count++;
                }
            }
            return count;
        };

        size_t reorders_count_fused = get_reorders_count(fused);
        size_t reorders_count_not_fused = get_reorders_count(not_fused);

        std::stringstream description;
        description << std::endl << "not fused: " << std::endl;
        for (auto i : not_fused.get_primitives_info()) {
            description << "  " << i.original_id << " " << i.kernel_id << std::endl;
        }
        description << "fused: " << std::endl;
        for (auto i : fused.get_primitives_info()) {
            description << "  " << i.original_id << " " << i.kernel_id << std::endl;
        }
        SCOPED_TRACE(description.str());
        // Subtract reorders count to handle execution in different layouts when input/output reorders can be added in the graph
        ASSERT_EQ(fused.get_executed_primitives().size() - (count_reorder ? 0 : reorders_count_fused), p.expected_fused_primitives);
        ASSERT_EQ(not_fused.get_executed_primitives().size() - (count_reorder ? 0 : reorders_count_not_fused), p.expected_not_fused_primitives);
        ASSERT_EQ(outputs_ref.size(), outputs_fused.size());
        ASSERT_EQ(outputs_ref.size(), size_t(1));

        auto output_not_fused_prim = outputs_ref.begin()->second.get_memory();
        auto output_fused_prim = outputs_fused.begin()->second.get_memory();
        if (output_not_fused_prim->get_layout().data_type == data_types::f32) {
            cldnn::mem_lock<float> ref(output_not_fused_prim, get_test_stream());
            cldnn::mem_lock<float> output_ptr(output_fused_prim, get_test_stream());
            for (size_t i = 0; i < output_fused_prim->get_layout().count(); i++) {
                ASSERT_NEAR(ref[i], output_ptr[i], tolerance) << "i = " << i;
            }
        } else {
            cldnn::mem_lock<int16_t> ref(output_not_fused_prim, get_test_stream());
            cldnn::mem_lock<int16_t> output_ptr(output_fused_prim, get_test_stream());
            for (size_t i = 0; i < output_fused_prim->get_layout().count(); i++) {
                ASSERT_NEAR(float16_to_float32(ref[i]), float16_to_float32(output_ptr[i]), tolerance) << "i = " << i;
            }
        }
    }

    void check_fusions_correctness(network& network_fused, std::map<std::string, std::vector<std::string>> expected_fused_primitives_ids = {}) {
        if (expected_fused_primitives_ids.size()) {
            auto primitives_info = network_fused.get_primitives_info();
            for (auto& prim : expected_fused_primitives_ids) {
                auto info = std::find_if(primitives_info.begin(), primitives_info.end(),
                                         [&prim](const primitive_info& info) -> bool { return info.original_id == prim.first; });
                if (info != primitives_info.end()) {
                    auto fused_primitives = info->c_fused_ids;
                    for (auto& expected_fused_prim : prim.second)
                        if (std::find(fused_primitives.begin(), fused_primitives.end(), expected_fused_prim) == fused_primitives.end())
                            FAIL() << "Couldn't find requested fused primitive id " + prim.first;
                } else {
                    FAIL() << "Couldn't find requested primitive id " + prim.first;
                }
            }
        }
    }

    cldnn::memory::ptr get_mem(cldnn::layout l) {
        auto prim = engine.allocate_memory(l);
        tensor s = l.size;
        if (l.data_type == data_types::bin) {
            VF<int32_t> rnd_vec = generate_random_1d<int32_t>(s.count() / 32, min_random, max_random);
            set_values(prim, rnd_vec);
        } else if (l.data_type == data_types::i8 || l.data_type == data_types::u8) {
            VF<uint8_t> rnd_vec = generate_random_1d<uint8_t>(s.count(), min_random, max_random);
            set_values(prim, rnd_vec);
        } else if (l.data_type == data_types::f16) {
            VF<uint16_t> rnd_vec = generate_random_1d<uint16_t>(s.count(), -1, 1);
            set_values(prim, rnd_vec);
        } else {
            VF<float> rnd_vec = generate_random_1d<float>(s.count(), -1, 1);
            set_values(prim, rnd_vec);
        }

        return prim;
    }

    cldnn::memory::ptr get_mem(cldnn::layout l, float fill_value) {
        auto prim = engine.allocate_memory(l);
        tensor s = l.size;
        if (l.data_type == data_types::bin) {
            VF<int32_t> rnd_vec(s.count() / 32, static_cast<int32_t>(fill_value));
            set_values(prim, rnd_vec);
        } else if (l.data_type == data_types::f16) {
            VF<uint16_t> rnd_vec(s.count(), float32_to_float16(fill_value));
            set_values(prim, rnd_vec);
        } else if (l.data_type == data_types::f32) {
            VF<float> rnd_vec(s.count(), fill_value);
            set_values(prim, rnd_vec);
        } else if (l.data_type == data_types::u8) {
            VF<uint8_t> rnd_vec(s.count(), static_cast<uint8_t>(fill_value));
            set_values(prim, rnd_vec);
        } else if (l.data_type == data_types::i8) {
            VF<int8_t> rnd_vec(s.count(), static_cast<int8_t>(fill_value));
            set_values(prim, rnd_vec);
        } else {
            throw std::runtime_error("get_mem: Unsupported precision");
        }

        return prim;
    }

    cldnn::memory::ptr get_repeatless_mem(cldnn::layout l, int min, int max) {
        auto prim = engine.allocate_memory(l);
        tensor s = l.size;
        if (l.data_type == data_types::f32) {
            VF<float> rnd_vec = generate_random_norepetitions_1d<float>(s.count(), min, max);
            set_values(prim, rnd_vec);
        } else if (l.data_type == data_types::f16) {
            VF<FLOAT16> rnd_vec = generate_random_norepetitions_1d<FLOAT16>(s.count(), min, max);
            set_values(prim, rnd_vec);
        } else if (l.data_type == data_types::i8) {
            VF<int8_t> rnd_vec = generate_random_norepetitions_1d<int8_t>(s.count(), min, max);
            set_values(prim, rnd_vec);
        }
        else if (l.data_type == data_types::bin) {
            VF<int32_t> rnd_vec = generate_random_norepetitions_1d<int32_t>(s.count(), min, max);
            set_values(prim, rnd_vec);
        }

        return prim;
    }

    cldnn::memory::ptr get_mem(cldnn::layout l, int min, int max) {
        auto prim = engine.allocate_memory(l);
        tensor s = l.size;
        if (l.data_type == data_types::f32) {
            VF<float> rnd_vec = generate_random_1d<float>(s.count(), min, max);
            set_values(prim, rnd_vec);
        } else if (l.data_type == data_types::f16) {
            VF<FLOAT16> rnd_vec = generate_random_1d<FLOAT16>(s.count(), min, max);
            set_values(prim, rnd_vec);
        } else if (l.data_type == data_types::i8) {
            VF<int8_t> rnd_vec = generate_random_1d<int8_t>(s.count(), min, max);
            set_values(prim, rnd_vec);
        } else if (l.data_type == data_types::u8) {
            VF<uint8_t> rnd_vec = generate_random_1d<uint8_t>(s.count(), min, max);
            set_values(prim, rnd_vec);
        } else if (l.data_type == data_types::bin) {
            VF<int32_t> rnd_vec = generate_random_1d<int32_t>(s.count() / 32, min, max);
            set_values(prim, rnd_vec);
        }

        return prim;
    }

    layout get_output_layout(T& p) {
        return layout{ p.data_type, p.input_format, p.out_shape };
    }

    layout get_weights_layout(T& p, const int32_t /* split */ = 1) {
        cldnn::tensor weights_tensor;
        if (p.groups == 1) {
            weights_tensor = cldnn::tensor(batch(p.out_shape.feature[0]), feature(p.in_shape.feature[0]),
                                           spatial(p.kernel.spatial[0], p.kernel.spatial[1], p.kernel.spatial[2]));
        } else {
            weights_tensor = cldnn::tensor(group(p.groups), batch(p.out_shape.feature[0] / p.groups), feature(p.in_shape.feature[0] / p.groups),
                                           spatial(p.kernel.spatial[0], p.kernel.spatial[1], p.kernel.spatial[2]));
        }
        return layout{p.weights_type, p.weights_format, weights_tensor};
    }

    layout get_weights_layout(T& p, const int32_t /* split */, cldnn::format f) {
        cldnn::tensor weights_tensor;
        weights_tensor = cldnn::tensor(batch(p.out_shape.feature[0]), feature(static_cast<int32_t>(p.in_shape.feature[0] / p.groups)),
                                       spatial(p.kernel.spatial[0], p.kernel.spatial[1], p.kernel.spatial[2]));
        return layout{p.weights_type, f, weights_tensor};
    }

    layout get_bias_layout(T& p) {
        return layout{ p.default_type, format::bfyx, tensor{1, p.out_shape.feature[0], 1, 1} };
    }

    layout get_weights_zp_layout(T& p) {
        return layout{ p.weights_type, p.default_format, tensor{p.out_shape.feature[0], 1, 1, 1} };
    }

    layout get_activations_zp_layout(T& p) {
        return layout{ p.data_type, p.default_format, tensor{1, p.in_shape.feature[0], 1, 1} };
    }

    layout get_single_element_layout(T& p) {
        return layout{ p.default_type, p.default_format, tensor{1, 1, 1, 1} };
    }

    template <class... Args>
    void create_topologies(Args const&... args) {
        topology_fused.add(args...);
        topology_non_fused.add(args...);
    }

    template <class... Args>
    void add_topologies(Args const&... args) {
        topology_fused.add(args...);
        topology_non_fused.add(args...);
    }
};
