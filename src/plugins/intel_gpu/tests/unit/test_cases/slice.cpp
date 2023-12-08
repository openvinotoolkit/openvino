// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/slice.hpp>
#include <intel_gpu/primitives/data.hpp>

#include <random>
#include <algorithm>
#include <vector>

using namespace cldnn;
using namespace ::tests;

namespace {

template<typename T>
class SliceTest : public ::testing::Test {
public:
    static std::vector<T> GenInput(int size) {
        std::vector<T> result;
        for (int i = 0; i < size; i++)
            result.push_back(i);
        return result;
    }

    void execute(bool is_caching_test) {
        assert(input_shape_.size() == 4 || input_shape_.size() == 5);
        format input_format = input_shape_.size() == 4 ? format::bfyx : format::bfzyx;
        layout data_layout ( input_type_, input_format, tensor{input_shape_} );
        std::vector<T> input_vals = GenInput(static_cast<int>(data_layout.get_linear_size()));
        memory::ptr input = engine_.allocate_memory(data_layout);
        set_values(input, input_vals);
        topology topology;
        topology.add(input_layout("input", input->get_layout()));
        topology.add(data("start", start_));
        topology.add(data("stop", stop_));
        topology.add(data("step", step_));
        std::vector<input_info> inputs { input_info("input"), input_info("start"), input_info("stop"), input_info("step") };
        if (axes_) {
            topology.add(data("axes", axes_));
            inputs.push_back(input_info("axes"));
        }
        topology.add(slice("slice", inputs));

        cldnn::network::ptr network = get_network(engine_, topology, get_test_default_config(engine_), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("input", input);

        auto outputs = network->execute();

        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "slice");

        auto output = outputs.at("slice").get_memory();

        cldnn::mem_lock<T> output_ptr(output, get_test_stream());

        ASSERT_EQ(output_ptr.size(), expected_output_.size());
        for (size_t i = 0; i < output_ptr.size(); ++i)
            ASSERT_TRUE(are_equal(expected_output_[i], output_ptr[i], 2e-3));
    }

    data_types DataType() const;

protected:
    engine& engine_ = get_test_engine();
    std::vector<std::int32_t> input_shape_;
    data_types input_type_ {DataType()};
    memory::ptr start_;
    memory::ptr stop_;
    memory::ptr step_;
    memory::ptr axes_;
    std::vector<std::int32_t> output_shape_;
    std::vector<T> expected_output_;
};

template<>
data_types SliceTest<float>::DataType() const {return data_types::f32;}

template<>
data_types SliceTest<int>::DataType() const { return data_types::i32; }

template<>
data_types SliceTest<long long>::DataType() const { return data_types::i64; }

using testing::Types;
typedef Types<float, int, long long> DataTypes;
TYPED_TEST_SUITE(SliceTest, DataTypes);

TYPED_TEST(SliceTest, bfyx_positive_step) {
    this->input_shape_ = { 1, 2, 100, 12 };
    this->start_ = this->engine_.allocate_memory({ data_types::i64, format::bfyx, { 4, 1, 1, 1 } });
    set_values<int64_t>(this->start_, {0, 1, 0, 1});
    this->stop_ = this->engine_.allocate_memory({ data_types::i64, format::bfyx, { 4, 1, 1, 1 } });
    set_values<int64_t>(this->stop_, { 1, 2, 5, 100 });
    this->step_ = this->engine_.allocate_memory({ data_types::i64, format::bfyx, { 4, 1, 1, 1 } });
    set_values<int64_t>(this->step_, { 1, 1, 1, 10 });
    this->output_shape_ = { 1, 1, 5, 10 };
    this->expected_output_ = {
            1201, 1211, 1221, 1231, 1241, 1301, 1311, 1321, 1331, 1341,
            1401, 1411, 1421, 1431, 1441, 1501, 1511, 1521, 1531, 1541,
            1601, 1611, 1621, 1631, 1641, 1701, 1711, 1721, 1731, 1741,
            1801, 1811, 1821, 1831, 1841, 1901, 1911, 1921, 1931, 1941,
            2001, 2011, 2021, 2031, 2041, 2101, 2111, 2121, 2131, 2141
    };
    this->execute(false);
}

TYPED_TEST(SliceTest, bfyx_negative_step) {
    this->input_shape_ = { 1, 2, 100, 12 };
    this->start_ = this->engine_.allocate_memory({ data_types::i64, format::bfyx, { 4, 1, 1, 1 } });
    set_values<int64_t>(this->start_, { 0, 1, 5, 100 });
    this->stop_ = this->engine_.allocate_memory({ data_types::i64, format::bfyx, { 4, 1, 1, 1 } });
    set_values<int64_t>(this->stop_, {1, 0, 0, 1});
    this->step_ = this->engine_.allocate_memory({ data_types::i64, format::bfyx, { 4, 1, 1, 1 } });
    set_values<int64_t>(this->step_, { 1, -1, -1, -10 });
    this->output_shape_ = { 1, 1, 5, 10 };
    this->expected_output_ = {
            1799, 1789, 1779, 1769, 1759, 1699, 1689, 1679, 1669, 1659,
            1599, 1589, 1579, 1569, 1559, 1499, 1489, 1479, 1469, 1459,
            1399, 1389, 1379, 1369, 1359, 1299, 1289, 1279, 1269, 1259,
            1199, 1189, 1179, 1169, 1159, 1099, 1089, 1079, 1069, 1059,
             999,   989,  979, 969,  959,  899,  889,  879,  869,  859
    };
    this->execute(false);
}

TYPED_TEST(SliceTest, bfzyx) {
    this->input_shape_ = { 2, 3, 10, 12, 5 };
    this->start_ = this->engine_.allocate_memory({ data_types::i64, format::bfzyx, { 5, 1, 1, 1 } });
    set_values<int64_t>(this->start_, { 0, 0, 0, 0, 0 });
    this->stop_ = this->engine_.allocate_memory({ data_types::i64, format::bfzyx, { 5, 1, 1, 1 } });
    set_values<int64_t>(this->stop_, {1, 2, 2, 2, 2});
    this->step_ = this->engine_.allocate_memory({ data_types::i64, format::bfzyx, { 5, 1, 1, 1 } });
    set_values<int64_t>(this->step_, { 1, 1, 1, 1, 1 });
    this->output_shape_ = { 1, 2, 2, 2, 2 };
    this->expected_output_ = {
              0,   1,  10,  11, 120, 121, 130, 131,
            600, 601, 610, 611, 720, 721, 730, 731
    };
    this->execute(false);
}

#ifdef RUN_ALL_MODEL_CACHING_TESTS
TYPED_TEST(SliceTest, bfyx_positive_step_cached) {
    this->input_shape_ = { 1, 2, 100, 12 };
    this->start_ = this->engine_.allocate_memory({ data_types::i64, format::bfyx, { 4, 1, 1, 1 } });
    set_values<int64_t>(this->start_, {0, 1, 0, 1});
    this->stop_ = this->engine_.allocate_memory({ data_types::i64, format::bfyx, { 4, 1, 1, 1 } });
    set_values<int64_t>(this->stop_, { 1, 2, 5, 100 });
    this->step_ = this->engine_.allocate_memory({ data_types::i64, format::bfyx, { 4, 1, 1, 1 } });
    set_values<int64_t>(this->step_, { 1, 1, 1, 10 });
    this->output_shape_ = { 1, 1, 5, 10 };
    this->expected_output_ = {
            1201, 1211, 1221, 1231, 1241, 1301, 1311, 1321, 1331, 1341,
            1401, 1411, 1421, 1431, 1441, 1501, 1511, 1521, 1531, 1541,
            1601, 1611, 1621, 1631, 1641, 1701, 1711, 1721, 1731, 1741,
            1801, 1811, 1821, 1831, 1841, 1901, 1911, 1921, 1931, 1941,
            2001, 2011, 2021, 2031, 2041, 2101, 2111, 2121, 2131, 2141
    };
    this->execute(true);
}

TYPED_TEST(SliceTest, bfyx_negative_step_cached) {
    this->input_shape_ = { 1, 2, 100, 12 };
    this->start_ = this->engine_.allocate_memory({ data_types::i64, format::bfyx, { 4, 1, 1, 1 } });
    set_values<int64_t>(this->start_, { 0, 1, 5, 100 });
    this->stop_ = this->engine_.allocate_memory({ data_types::i64, format::bfyx, { 4, 1, 1, 1 } });
    set_values<int64_t>(this->stop_, {1, 0, 0, 1});
    this->step_ = this->engine_.allocate_memory({ data_types::i64, format::bfyx, { 4, 1, 1, 1 } });
    set_values<int64_t>(this->step_, { 1, -1, -1, -10 });
    this->output_shape_ = { 1, 1, 5, 10 };
    this->expected_output_ = {
            1799, 1789, 1779, 1769, 1759, 1699, 1689, 1679, 1669, 1659,
            1599, 1589, 1579, 1569, 1559, 1499, 1489, 1479, 1469, 1459,
            1399, 1389, 1379, 1369, 1359, 1299, 1289, 1279, 1269, 1259,
            1199, 1189, 1179, 1169, 1159, 1099, 1089, 1079, 1069, 1059,
             999,   989,  979, 969,  959,  899,  889,  879,  869,  859
    };
    this->execute(true);
}
#endif
TYPED_TEST(SliceTest, bfzyx_cached) {
    this->input_shape_ = { 2, 3, 10, 12, 5 };
    this->start_ = this->engine_.allocate_memory({ data_types::i64, format::bfzyx, { 5, 1, 1, 1 } });
    set_values<int64_t>(this->start_, { 0, 0, 0, 0, 0 });
    this->stop_ = this->engine_.allocate_memory({ data_types::i64, format::bfzyx, { 5, 1, 1, 1 } });
    set_values<int64_t>(this->stop_, {1, 2, 2, 2, 2});
    this->step_ = this->engine_.allocate_memory({ data_types::i64, format::bfzyx, { 5, 1, 1, 1 } });
    set_values<int64_t>(this->step_, { 1, 1, 1, 1, 1 });
    this->output_shape_ = { 1, 2, 2, 2, 2 };
    this->expected_output_ = {
              0,   1,  10,  11, 120, 121, 130, 131,
            600, 601, 610, 611, 720, 721, 730, 731
    };
    this->execute(true);
}

} // anonymous namespace
