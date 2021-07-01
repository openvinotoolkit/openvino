// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <cldnn/primitives/input_layout.hpp>
#include <cldnn/primitives/convolution.hpp>
#include <cldnn/primitives/data.hpp>

#include <iostream>
#include <fstream>
#include <string>

namespace {

enum class cache_version {
    version_1,
    version_1_2,  // version 1 cache, but version 2 file
    version_2,
    version_2_invalid,
    version_2_from_1,
    version_2_empty
};

std::string reference_impl_name = "fused_conv_eltwise_gpu_ref";
std::string eus_marker = "__EUs__";

std::string cache_v1 =
R"__a({
    "__EUs__": {
        "18283230515392601293": ["fused_conv_eltwise_gpu_ref", 0]
    }
})__a";

std::string cache_v1_2 =
R"__a({
    "version_2": {
    },
    "version_1": {
        "__EUs__": {
            "18283230515392601293": ["fused_conv_eltwise_gpu_ref", 0]
        }
    }
})__a";

std::string cache_v2 =
R"__a({
    "version_2": {
        "__EUs__": {
            "CONVOLUTION": {
                "F32_BFYX_v3_p0_0_v3_p0_0_v16_p0_0_v1_p0_0;F32_BFYX_v3_p0_0_v3_p0_0_v16_p0_0_v1_p0_0;1_1_1;1_1_1;1_1_1;0_0_0;1;1": ["fused_conv_eltwise_gpu_ref", 0]
            }
        }
    }
})__a";

std::string cache_v2_from_v1 =
R"__a({
    "version_2": {
        "__EUs__": {
            "CONVOLUTION": {
                "F32_BFYX_v3_p0_0_v3_p0_0_v16_p0_0_v1_p0_0;F32_BFYX_v3_p0_0_v3_p0_0_v16_p0_0_v1_p0_0;1_1_1;1_1_1;1_1_1;0_0_0;1;1": ["fused_conv_eltwise_gpu_ref", 0]
            }
        }
    },
    "version_1": {
        "__EUs__": {}
    }
})__a";

std::string cache_v2_invalid =
R"__a({
    "version_2": {
        "__EUs__": {
            "CONVOLUTION": {
                "F32_BFYX_v3_p0_0_v3_p0_0_v16_p0_0_v1_p0_0;F32_BFYX_v3_p0_0_v3_p0_0_v16_p0_0_v1_p0_0;1_1_1;1_1_1;1_1_1;0_0_0;1;1": ["non_existent", 0]
            }
        }
    }
})__a";

std::string cache_v2_empty =
R"__a({
    "version_2": {
        "__EUs__": {
            "CONVOLUTION": {}
        }
    }
})__a";

std::string get_cache_version(cache_version version) {
    std::string cache;
    switch (version) {
    case cache_version::version_1:
        cache = cache_v1;
        break;
    case cache_version::version_1_2:
        cache = cache_v1_2;
        break;
    case cache_version::version_2:
        cache = cache_v2;
        break;
    case cache_version::version_2_invalid:
        cache = cache_v2_invalid;
        break;
    case cache_version::version_2_from_1:
        cache = cache_v2_from_v1;
        break;
    case cache_version::version_2_empty:
        cache = cache_v2_empty;
        break;
    default:
        throw std::invalid_argument("invalid cache version");
    }
    return cache;
}

std::string get_temporary_cache_file() {
    static int i = 0;
    std::string tmp_cache_file = "tmp_cldnn_test_cache_" + std::to_string(i) + ".json";
    i += 1;
    return tmp_cache_file;
}

template <typename T>
void replace(std::string& text, const std::string& replaced, T replacement) {
    auto it = text.find(replaced);
    while (it != std::string::npos) {
        text.replace(it, replaced.length(), std::to_string(replacement));
        it = text.find(replaced);
    }
}

void write(const std::string& filename, const std::string& text) {
    std::ofstream file;
    file.open(filename);
    if (!file.is_open())
        throw std::runtime_error("Could not open file " + filename);
    file << text;
    file.close();
    if (!file) {
        throw std::runtime_error("Failure writing to file " + filename);
    }
}

std::string read(const std::string& filename) {
    std::stringstream ss;
    std::ifstream file;
    file.open(filename);
    if (!file.is_open())
        throw std::runtime_error("Could not open file " + filename);

    ss << file.rdbuf();
    file.close();
    if (!file) {
        throw std::runtime_error("Failure reading from file " + filename);
    }
    return ss.str();
}

void remove(const std::string& filename) {
    std::remove(filename.c_str());
}

class cache_test_helper {
public:
    cache_test_helper(cldnn::engine& engine, cache_version v)
        : _engine(engine)
        , _mode(cldnn::tuning_mode::tuning_disabled)
        , cache_filename(get_temporary_cache_file())
    {
        auto cache = get_cache_version(v);
        auto eus = engine.get_device_info().execution_units_count;
        replace(cache, eus_marker, eus);

        write(cache_filename, cache);
    }

    virtual ~cache_test_helper() {
        remove(cache_filename);
    }

    cache_test_helper& with_mode(cldnn::tuning_mode mode) {
        _mode = mode;
        return *this;
    }

    cache_test_helper& expect_cache(cache_version version) {
        compare_cache = version;
        return *this;
    }

    cache_test_helper& expect_implementation(std::string implementation) {
        compare_implementation = implementation;
        return *this;
    }

    cache_test_helper& expect_implementation_not(std::string implementation) {
        compare_implementation = implementation;
        compare_implementation.not_equal = true;
        return *this;
    }

    void test() {
        auto w_mem = _engine.allocate_memory(cldnn::layout(cldnn::data_types::f32, cldnn::format::bfyx, { 16, 16, 1, 1 }));
        auto topology = cldnn::topology(
            cldnn::input_layout("input", cldnn::layout(cldnn::data_types::f32, cldnn::format::bfyx, { 1, 16, 3, 3 })),
            cldnn::data("weights", w_mem),
            cldnn::convolution("conv", "input", { "weights" })
        );

        auto tune_conf = cldnn::tuning_config_options();
        tune_conf.cache_file_path = cache_filename;
        tune_conf.mode = _mode;
        auto build_opts = cldnn::build_options(
            cldnn::build_option::tuning_config(tune_conf),
            cldnn::build_option::optimize_data(true)
        );
        auto network = cldnn::network(_engine, topology, build_opts);
        auto in_mem = _engine.allocate_memory(cldnn::layout(cldnn::data_types::f32, cldnn::format::bfyx, { 1, 16, 3, 3 }));
        network.set_input_data("input", in_mem);
        network.execute();

        if (compare_implementation.compare) {
            std::string exec_impl;
            for (auto& info : network.get_primitives_info()) {
                if (info.original_id == "conv") {
                    exec_impl = info.kernel_id;
                    break;
                }
            }
            if (compare_implementation.not_equal) {
                EXPECT_NE(exec_impl, compare_implementation.value);
            } else {
                EXPECT_EQ(exec_impl, compare_implementation.value);
            }
        }

        if (compare_cache.compare) {
            auto cache = read(cache_filename);
            auto expected_cache = get_cache_version(compare_cache.value);
            auto eus = _engine.get_device_info().execution_units_count;
            replace(expected_cache, eus_marker, eus);

            EXPECT_EQ(cache, expected_cache);
        }
    }

private:
    template <typename T>
    struct optional_compare {
        bool compare;
        bool not_equal;
        T value;

        optional_compare() : compare(false) {}
        optional_compare(T v) : compare(true), not_equal(false), value(v) {}
        optional_compare(T v, bool neq) : compare(true), not_equal(neq), value(v) {}
    };

    cldnn::engine& _engine;

    cldnn::tuning_mode _mode;

    std::string cache_filename;

    optional_compare<cache_version> compare_cache;
    optional_compare<std::string> compare_implementation;
};

}  // namespace

class cache_version_test : public testing::TestWithParam<cache_version> {
public:
    static std::string to_string(const testing::TestParamInfo<cache_version>& param) {
        std::string result;
        switch (param.param) {
        case cache_version::version_1:
            result = "version_1";
            break;
        case cache_version::version_1_2:
            result = "version_1_2";
            break;
        case cache_version::version_2:
            result = "version_2";
            break;
        case cache_version::version_2_invalid:
            result = "version_2_invalid";
            break;
        case cache_version::version_2_from_1:
            result = "version_2_from_1";
            break;
        case cache_version::version_2_empty:
            result = "version_2_empty";
            break;
        default:
            result = std::to_string(static_cast<int>(param.param));
            break;
        }
        return result;
    }
};

TEST(cache_test, no_cache_baseline) {
    SCOPED_TRACE("default implementation same as reference, cache tests may provide invalid pass");
    auto& engine = tests::get_test_engine();
    auto helper = cache_test_helper(engine, cache_version::version_2);

    helper.with_mode(cldnn::tuning_mode::tuning_disabled)
        .expect_implementation_not(reference_impl_name)
        .test();
}

TEST_P(cache_version_test, use_only) {
    auto version = GetParam();
    auto& engine = tests::get_test_engine();

    cache_test_helper helper(engine, version);
    helper.with_mode(cldnn::tuning_mode::tuning_use_cache)
        .expect_implementation(reference_impl_name)
        .expect_cache(version)
        .test();
}

TEST_P(cache_version_test, update) {
    auto version = GetParam();
    auto ex_version = cache_version::version_2;
    if (version != cache_version::version_2) {
        ex_version = cache_version::version_2_from_1;
    }

    auto& engine = tests::get_test_engine();

    cache_test_helper helper(engine, version);
    helper.with_mode(cldnn::tuning_mode::tuning_use_and_update)
        .expect_implementation(reference_impl_name)
        .expect_cache(ex_version)
        .test();
}

INSTANTIATE_TEST_SUITE_P(
    smoke,
    cache_version_test,
    testing::Values(cache_version::version_1, cache_version::version_1_2, cache_version::version_2),
    cache_version_test::to_string);

TEST(cache_test, remove_invalid) {
    auto& engine = tests::get_test_engine();

    cache_test_helper helper(engine, cache_version::version_2_invalid);
    helper.with_mode(cldnn::tuning_mode::tuning_use_and_update)
        .expect_implementation_not(reference_impl_name)
        .expect_cache(cache_version::version_2_empty)
        .test();
}
