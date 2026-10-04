#include "shared_test_classes/base/ov_subgraph.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/result.hpp"

#include <cstring>
#include <sstream>

namespace {

class DividePythonDivisionTest : public testing::WithParamInterface<std::tuple<ov::element::Type, bool>>,
                                 virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<std::tuple<ov::element::Type, bool>>& obj) {
        const auto& [et, pythondiv] = obj.param;
        std::ostringstream result;
        result << "ET=" << et << "_pythondiv=" << pythondiv;
        return result.str();
    }

protected:
    void SetUp() override {
        const auto& [et, pythondiv] = GetParam();
        targetDevice = ov::test::utils::DEVICE_GPU;

        auto lhs = std::make_shared<ov::op::v0::Parameter>(et, ov::Shape{4});
        auto rhs = std::make_shared<ov::op::v0::Parameter>(et, ov::Shape{4});
        auto divide = std::make_shared<ov::op::v1::Divide>(lhs, rhs, pythondiv);
        auto result = std::make_shared<ov::op::v0::Result>(divide);
        function = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{lhs, rhs}, "divide");

        ov::test::InputShape shape{ov::PartialShape{4}, {ov::Shape{4}}};
        init_input_shapes({shape, shape});
    }

    void generate_inputs(const std::vector<ov::Shape>& target_input_static_shapes) override {
        inputs.clear();
        const auto& et = std::get<0>(GetParam());

        std::vector<ov::Tensor> tensors;
        if (et == ov::element::i16) {
            const std::vector<int16_t> l{-5, 5, -5, 5};
            const std::vector<int16_t> r{2, -2, -2, 2};
            tensors.emplace_back(ov::Tensor{et, target_input_static_shapes[0]});
            std::memcpy(tensors[0].data(), l.data(), l.size() * sizeof(int16_t));
            tensors.emplace_back(ov::Tensor{et, target_input_static_shapes[1]});
            std::memcpy(tensors[1].data(), r.data(), r.size() * sizeof(int16_t));
        } else {
            const std::vector<int32_t> l{-5, 5, -5, 5};
            const std::vector<int32_t> r{2, -2, -2, 2};
            tensors.emplace_back(ov::Tensor{et, target_input_static_shapes[0]});
            std::memcpy(tensors[0].data(), l.data(), l.size() * sizeof(int32_t));
            tensors.emplace_back(ov::Tensor{et, target_input_static_shapes[1]});
            std::memcpy(tensors[1].data(), r.data(), r.size() * sizeof(int32_t));
        }

        const auto params = function->get_parameters();
        inputs.insert({params[0], tensors[0]});
        inputs.insert({params[1], tensors[1]});
    }

    void validate() override {
        const auto& [et, pythondiv] = GetParam();
        const auto actual = get_plugin_outputs();
        ASSERT_EQ(actual.size(), 1);
        const auto& out_tensor = actual[0];

        auto check = [&](auto sentinel) {
            using T = decltype(sentinel);
            const auto* data = out_tensor.data<T>();
            std::vector<T> expected;
            if (pythondiv) {
                expected = {T(-3), T(-3), T(2), T(2)};
            } else {
                expected = {T(-2), T(-2), T(2), T(2)};
            }
            for (size_t i = 0; i < expected.size(); ++i) {
                ASSERT_EQ(expected[i], data[i]);
            }
        };

        if (et == ov::element::i16) {
            check(int16_t(0));
        } else {
            check(int32_t(0));
        }
    }
};

TEST_P(DividePythonDivisionTest, IntegerDividePropagatesPythondiv) {
    run();
}

}

INSTANTIATE_TEST_SUITE_P(smoke_IntegerDivide,
                         DividePythonDivisionTest,
                         ::testing::Combine(::testing::Values(ov::element::i16, ov::element::i32),
                                            ::testing::Values(true, false)),
                         DividePythonDivisionTest::getTestCaseName);
