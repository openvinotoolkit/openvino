// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/symbolic_transformations/dereshape_matmul.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/util/op_types.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov;
using namespace ov::op;
using namespace std;

namespace {
/* Helps to organize dimension representation in the following tests:
 * 1. Creates requested amount of dimensions
 * 2. Sets unique symbols for them automatically
 * 3. Creates value representation of the dimension via creating Parameter->Shape->Gather subgraph
 * 4. Gives access to dimension and its value representation via operator[]
 * 5. Gives access to utility Parameter via get_parameter -- only used for ov::Model creation in tests
 * */
class DimensionTestHelper {
public:
    struct DimensionWithOutput {
        Dimension dim;
        Output<Node> source;
    };

    explicit DimensionTestHelper(const size_t& num_dims) {
        auto dimensions = PartialShape::dynamic(Rank(num_dims));
        set_shape_symbols(dimensions);
        parameter = make_shared<v0::Parameter>(element::f32, dimensions);
        for (size_t i = 0; i < num_dims; ++i)
            m_map[i] = {dimensions[i], op::util::node_to_get_shape_value_of_indices_from_shape_source(parameter, {i})};
    }

    DimensionWithOutput operator[](size_t idx) const {
        return m_map.at(idx);
    }

    ov::PartialShape make_shape(const vector<size_t>& dim_indices) const {
        auto shape = PartialShape::dynamic(Rank(dim_indices.size()));
        for (size_t i = 0; i < dim_indices.size(); ++i)
            shape[i] = m_map.at(dim_indices[i]).dim;
        return shape;
    }

    shared_ptr<Node> make_reshape(const Output<Node>& source, const vector<size_t>& dims_indices) const {
        OutputVector sources(dims_indices.size());
        for (size_t i = 0; i < dims_indices.size(); ++i)
            sources[i] = m_map.at(dims_indices[i]).source;
        auto concat = make_shared<v0::Concat>(sources, 0);
        return make_shared<v1::Reshape>(source, concat, false);
    }

    std::shared_ptr<v0::Parameter> get_parameter() const {
        return parameter;
    }

private:
    std::shared_ptr<v0::Parameter> parameter;
    std::map<size_t, DimensionWithOutput> m_map;
};

size_t max_element(const vector<vector<size_t>>& vectors) {
    size_t current_max = 0;
    for (const auto& vector : vectors)
        current_max = max(current_max, *std::max_element(vector.begin(), vector.end()));
    return current_max;
}

shared_ptr<Node> reshape(const Output<Node>& source,
                         const vector<size_t>& dims_indices,
                         const DimensionTestHelper& helper) {
    OutputVector sources(dims_indices.size());
    for (size_t i = 0; i < dims_indices.size(); ++i)
        sources[i] = helper[dims_indices[i]].source;
    auto concat = make_shared<v0::Concat>(sources, 0);
    return make_shared<v1::Reshape>(source, concat, false);
}

void get_dims(const ov::Output<ov::Node>& source, const size_t& from, const size_t& to, ov::NodeVector& dims) {
    std::vector<size_t> non_constant_ids;
    for (size_t i = from; i < to; ++i) {
        auto node = ov::op::util::node_to_get_shape_value_of_indices_from_shape_source(source, {i});
        if (auto constant = ov::util::get_constant_from_source(node)) {
            node = constant;
        } else {
            non_constant_ids.push_back(i);
        }
        dims.push_back(node);
    }
}

ov::Output<ov::Node> get_target_shape_from_sources(const ov::Output<ov::Node>& batch_dims_source,
                                                   const ov::Output<ov::Node>& non_batch_dims_source) {
    ov::NodeVector dims;
    // batch dims here stand for MatMul batch dims -- leaving two last dims for Matrix Multiplication
    size_t num_batch_dims = batch_dims_source.get_partial_shape().size() - 2;
    get_dims(batch_dims_source, 0, num_batch_dims, dims);

    size_t non_batch_dims_start = non_batch_dims_source.get_partial_shape().size() - 2;
    get_dims(non_batch_dims_source, non_batch_dims_start, non_batch_dims_start + 2, dims);

    size_t num_non_const_nodes = 0;  // candidates for becoming a Constant -1 -- special value for Reshape pattern
    for (size_t curr_i = 0; curr_i + 1 < dims.size(); ++curr_i) {
        auto curr_node = dims[curr_i], next_node = dims[curr_i + 1];
        bool curr_is_const = ov::op::util::is_constant(curr_node), next_is_const = ov::op::util::is_constant(next_node);
        if (num_non_const_nodes == 0 && !curr_is_const && next_is_const) {
            curr_node = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
            curr_is_const = true;
            num_non_const_nodes += 1;
        }
        if (num_non_const_nodes == 0 && !next_is_const && curr_is_const) {
            next_node = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
            next_is_const = true;
            num_non_const_nodes += 1;
        }
        if (curr_is_const && next_is_const) {
            dims[curr_i] = nullptr;
            dims[curr_i + 1] = ov::op::util::make_try_fold<ov::op::v0::Concat>(ov::NodeVector{curr_node, next_node}, 0);
        }
    }
    dims.erase(std::remove_if(dims.begin(),
                              dims.end(),
                              [](const std::shared_ptr<ov::Node>& node) {
                                  return node == nullptr;
                              }),
               dims.end());
    auto target_shape = ov::op::util::make_try_fold<ov::op::v0::Concat>(dims, 0);
    return target_shape->output(0);
}

PartialShape make_concat_input_pshape(const DimensionTestHelper& dims, const vector<size_t>& dims_indices) {
    auto another_pshape = dims.make_shape(dims_indices);
    size_t rank = dims_indices.size();
    // To reduce test graph we avoid changing Concat axis dimension with this Concat
    another_pshape[rank - 1] = Dimension(0);
    return another_pshape;
}

static std::ostream& operator<<(std::ostream& os, const vector<size_t>& vals) {
    bool first = true;
    for (const auto& val : vals) {
        if (!first)
            os << "_";
        first = false;
        os << val;
    }
    return os;
}
}  // namespace

using DeReshapeMatMulParameters =
    tuple<tuple<vector<size_t>, vector<size_t>, vector<size_t>, vector<size_t>, vector<size_t>>,
          size_t,
          size_t,
          size_t>;

class DeReshapeMatMulTest : public TransformationTestsF, public testing::WithParamInterface<DeReshapeMatMulParameters> {
public:
    void SetUp() override {
        TransformationTestsF::SetUp();
        const auto& params = std::get<0>(GetParam());

        const auto& lhs_shape_idx = std::get<0>(params);
        const auto& lhs_reshape_idx = std::get<1>(params);
        const auto& rhs_shape_idx = std::get<2>(params);
        const auto& rhs_reshape_idx = std::get<3>(params);
        const auto& out_reshape_idx = std::get<4>(params);

        // 0 - no bea, 1 - lhs, 2 - rhs, 3 - lhs and rhs
        const size_t& bea_scalar_mode = std::get<1>(GetParam());

        // 0 - no concat
        // 10 - concat on lhs, reshape on 0 port
        // 11 - concat on lhs, reshape on 1 port
        // 20 - concat on rhs, reshape on 0 port
        // 21 - concat on rhs, reshape on 1 port
        // 300 - concat on both sizes, both reshapes on 0 port of concats
        // 301 - concat on both sizes, lhs reshape on 0 port, rhs reshape on 1 port
        // 310 - concat on both sizes, lhs reshape on 1 port, rhs reshape on 0 port
        // 311 - concat on both sizes, both reshapes on 1 port of concats
        const size_t& concat_mode = std::get<2>(GetParam());

        // 0 - no add, 1 - add has matmul on lhs, 2 - add has matmul on rhs
        const size_t& final_add_mode = std::get<3>(GetParam());

        const auto& max_idx =
            max_element({lhs_shape_idx, rhs_shape_idx, lhs_reshape_idx, rhs_reshape_idx, out_reshape_idx});
        const DimensionTestHelper dims(max_idx + 1);

        PartialShape lhs_original_pshape = dims.make_shape(lhs_shape_idx);
        PartialShape rhs_original_pshape = dims.make_shape(rhs_shape_idx);

        get_model(dims,
                  lhs_original_pshape,
                  rhs_original_pshape,
                  lhs_reshape_idx,
                  rhs_reshape_idx,
                  out_reshape_idx,
                  bea_scalar_mode,
                  concat_mode,
                  final_add_mode);
        manager.register_pass<pass::DeReshapeMatMul>();

        if (lhs_shape_idx.size() != 4 && lhs_reshape_idx.size() != 3 && final_add_mode != 0)
            return;  // check that for all those cases transformation doesn't do anything

        get_model_ref(dims,
                      lhs_original_pshape,
                      rhs_original_pshape,
                      lhs_reshape_idx,
                      rhs_reshape_idx,
                      bea_scalar_mode,
                      concat_mode,
                      final_add_mode);
    }

    void get_model(const DimensionTestHelper& dims,
                   const PartialShape& lhs_original_pshape,
                   const PartialShape& rhs_original_pshape,
                   const vector<size_t>& lhs_reshape_idx,
                   const vector<size_t>& rhs_reshape_idx,
                   const vector<size_t>& out_reshape_idx,
                   const size_t& bea_scalar_mode,
                   const size_t& concat_mode,
                   const size_t& final_add_mode) {
        ParameterVector inputs;
        OutputVector outputs;

        // LHS input of MatMul
        auto lhs_input = make_shared<v0::Parameter>(element::f32, lhs_original_pshape);
        auto lhs_output = dims.make_reshape(lhs_input, lhs_reshape_idx);

        if (set<size_t>{10, 11, 300, 301, 310, 311}.count(concat_mode)) {
            const auto& another_pshape = make_concat_input_pshape(dims, lhs_reshape_idx);
            const auto& another_input = make_shared<v0::Parameter>(element::f32, another_pshape);

            if (set<size_t>{10, 300, 301}.count(concat_mode)) {  // reshape on 0 port
                lhs_output = make_shared<v0::Concat>(OutputVector{lhs_output, another_input}, -1);
            } else if (set<size_t>{11, 310, 311}.count(concat_mode)) {  // reshape on 1 port
                lhs_output = make_shared<v0::Concat>(OutputVector{another_input, lhs_output}, -1);
            } else {
                ASSERT_TRUE(false) << "Unknown mode of concat: " << concat_mode;
            }
            inputs.push_back(another_input);
            outputs.emplace_back(lhs_output);
        }

        if (bea_scalar_mode == 1 || bea_scalar_mode == 3)
            lhs_output = make_shared<v1::Multiply>(lhs_output, v0::Constant::create(element::f32, {}, {0.125}));

        // RHS input of MatMul
        auto rhs_input = make_shared<v0::Parameter>(element::f32, rhs_original_pshape);
        auto rhs_output = dims.make_reshape(rhs_input, rhs_reshape_idx);

        if (set<size_t>{20, 21, 300, 301, 310, 311}.count(concat_mode)) {
            const auto& another_pshape = make_concat_input_pshape(dims, rhs_reshape_idx);
            const auto& another_input = make_shared<v0::Parameter>(element::f32, another_pshape);
            if (set<size_t>{20, 300, 310}.count(concat_mode)) {  // reshape on 0 port
                rhs_output = make_shared<v0::Concat>(OutputVector{rhs_output, another_input}, -1);
            } else if (set<size_t>{21, 301, 311}.count(concat_mode)) {  // reshape on 1 port
                rhs_output = make_shared<v0::Concat>(OutputVector{another_input, rhs_output}, -1);
            } else {
                ASSERT_TRUE(false) << "Unknown mode of concat: " << concat_mode;
            }
            inputs.push_back(another_input);
            outputs.emplace_back(rhs_output);
        }

        if (bea_scalar_mode == 2 || bea_scalar_mode == 3)
            rhs_output = make_shared<v1::Multiply>(rhs_output, v0::Constant::create(element::f32, {}, {0.125}));

        Output<Node> matmul = make_shared<v0::MatMul>(lhs_output, rhs_output);

        if (final_add_mode == 1)  // 1 - add has matmul on lhs
            matmul =
                make_shared<v1::Add>(matmul, v0::Constant::create(element::f32, Shape(lhs_reshape_idx.size(), 1), {1}));
        else if (final_add_mode == 2)  // 2 - add has matmul on rhs
            matmul =
                make_shared<v1::Add>(v0::Constant::create(element::f32, Shape(lhs_reshape_idx.size(), 1), {1}), matmul);

        auto output_reshape = reshape(matmul, out_reshape_idx, dims);

        inputs.push_back(dims.get_parameter());
        inputs.push_back(lhs_input);
        inputs.push_back(rhs_input);
        outputs.emplace_back(output_reshape);

        for (auto& output : outputs)
            output = std::make_shared<v1::Reshape>(output, v0::Constant::create(element::i32, {1}, {-1}), false);
        auto output = make_shared<v0::Concat>(outputs, 0);
        model = make_shared<Model>(output, inputs, "Tested model");
    }

    void get_model_ref(const DimensionTestHelper& dims,
                       const PartialShape& lhs_original_pshape,
                       const PartialShape& rhs_original_pshape,
                       const vector<size_t>& lhs_reshape_idx,
                       const vector<size_t>& rhs_reshape_idx,
                       const size_t& bea_scalar_mode,
                       const size_t& concat_mode,
                       const size_t& final_add_mode) {
        ParameterVector inputs;
        OutputVector outputs;

        // LHS input of MatMul
        auto lhs_input = make_shared<v0::Parameter>(element::f32, lhs_original_pshape);
        auto lhs_output = lhs_input->output(0);

        if (set<size_t>{10, 11, 300, 301, 310, 311}.count(concat_mode)) {
            const auto& another_pshape = make_concat_input_pshape(dims, lhs_reshape_idx);
            const auto& another_input = make_shared<v0::Parameter>(element::f32, another_pshape);

            auto target_shape_of_input = get_target_shape_from_sources(lhs_output, another_input);
            auto input_reshape = make_shared<v1::Reshape>(another_input, target_shape_of_input, false);

            if (set<size_t>{10, 300, 301}.count(concat_mode)) {  // reshape on 0 port
                lhs_output = make_shared<v0::Concat>(OutputVector{lhs_output, input_reshape}, -1);
            } else if (set<size_t>{11, 310, 311}.count(concat_mode)) {  // reshape on 1 port
                lhs_output = make_shared<v0::Concat>(OutputVector{input_reshape, lhs_output}, -1);
            } else {
                ASSERT_TRUE(false) << "Unknown mode of concat: " << concat_mode;
            }

            auto target_shape_of_output = get_target_shape_from_sources(input_reshape->input_value(0), lhs_output);
            auto output_reshape = make_shared<v1::Reshape>(lhs_output, target_shape_of_output, false);

            inputs.push_back(another_input);
            outputs.emplace_back(output_reshape);
        }

        if (bea_scalar_mode == 1 || bea_scalar_mode == 3)
            lhs_output = make_shared<v1::Multiply>(lhs_output, v0::Constant::create(element::f32, {}, {0.125}));

        // RHS input of MatMul
        auto rhs_input = make_shared<v0::Parameter>(element::f32, rhs_original_pshape);
        auto rhs_output = rhs_input->output(0);

        if (set<size_t>{20, 21, 300, 301, 310, 311}.count(concat_mode)) {
            const auto& another_pshape = make_concat_input_pshape(dims, rhs_reshape_idx);
            const auto& another_input = make_shared<v0::Parameter>(element::f32, another_pshape);

            auto target_shape_of_input = get_target_shape_from_sources(rhs_output, another_input);
            auto input_reshape = make_shared<v1::Reshape>(another_input, target_shape_of_input, false);

            if (set<size_t>{20, 300, 310}.count(concat_mode)) {  // reshape on 0 port
                rhs_output = make_shared<v0::Concat>(OutputVector{rhs_output, input_reshape}, -1);
            } else if (set<size_t>{21, 301, 311}.count(concat_mode)) {  // reshape on 1 port
                rhs_output = make_shared<v0::Concat>(OutputVector{input_reshape, rhs_output}, -1);
            } else {
                ASSERT_TRUE(false) << "Unknown mode of concat: " << concat_mode;
            }
            auto target_shape_of_output = get_target_shape_from_sources(input_reshape->input_value(0), rhs_output);
            auto output_reshape = make_shared<v1::Reshape>(rhs_output, target_shape_of_output, false);

            inputs.push_back(another_input);
            outputs.emplace_back(output_reshape);
        }

        if (bea_scalar_mode == 2 || bea_scalar_mode == 3)
            rhs_output = make_shared<v1::Multiply>(rhs_output, v0::Constant::create(element::f32, {}, {0.125}));

        Output<Node> matmul = make_shared<v0::MatMul>(lhs_output, rhs_output);

        if (final_add_mode > 0) {
            const auto original_add_in = v0::Constant::create(element::f32, Shape(lhs_reshape_idx.size(), 1), {1});
            auto divisor = ov::op::util::node_to_get_shape_value_of_indices_from_shape_source(lhs_input, {1});
            auto first_batch_dim =
                std::make_shared<ov::op::v1::Divide>(ov::op::v0::Constant::create(element::i64, {1}, {1}),
                                                     divisor,
                                                     true);
            auto minus_one = ov::op::v0::Constant::create(element::i64, {1}, {-1});
            auto non_batch_dims = ov::op::v0::Constant::create(element::i64, {2}, {1, 1});
            auto pattern =
                std::make_shared<ov::op::v0::Concat>(OutputVector{first_batch_dim, minus_one, non_batch_dims}, 0);
            auto other_input_reshape = op::util::make_try_fold<ov::op::v1::Reshape>(original_add_in, pattern, true);

            if (final_add_mode == 1) {  // 1 - add has matmul on lhs
                matmul = make_shared<v1::Add>(matmul, other_input_reshape);
            } else if (final_add_mode == 2) {  // 2 - add has matmul on rhs
                matmul = make_shared<v1::Add>(other_input_reshape, matmul);
            }
        }
        inputs.push_back(dims.get_parameter());
        inputs.push_back(lhs_input);
        inputs.push_back(rhs_input);
        outputs.emplace_back(matmul);

        for (auto& output : outputs)
            output = std::make_shared<v1::Reshape>(output, v0::Constant::create(element::i32, {1}, {-1}), false);
        auto output = make_shared<v0::Concat>(outputs, 0);

        model_ref = make_shared<Model>(output, inputs, "Reference model");
    }

    static std::string getTestCaseName(const testing::TestParamInfo<DeReshapeMatMulParameters>& obj) {
        vector<size_t> lhs_input_shape_indices, lhs_reshape_indices;
        vector<size_t> rhs_input_shape_indices, rhs_reshape_indices;
        vector<size_t> output_reshape_indices;
        size_t bea_scalar_mode, concat_mode, final_add_mode;

        tuple<vector<size_t>, vector<size_t>, vector<size_t>, vector<size_t>, vector<size_t>> tmp;

        std::tie(tmp, bea_scalar_mode, concat_mode, final_add_mode) = obj.param;
        std::tie(lhs_input_shape_indices,
                 lhs_reshape_indices,
                 rhs_input_shape_indices,
                 rhs_reshape_indices,
                 output_reshape_indices) = tmp;

        std::ostringstream result;
        result << "l_in_shape_idx=" << lhs_input_shape_indices << "_l_reshape_idx=" << lhs_reshape_indices
               << "_r_in_shape_idx=" << rhs_input_shape_indices << "_r_reshape_idx=" << rhs_reshape_indices
               << "_out_reshape_idx=" << output_reshape_indices << "_bea_scalar_mode=" << bea_scalar_mode
               << "_concat_mode=" << concat_mode << "_final_add_mode=" << final_add_mode;
        return result.str();
    }
};

const auto shape_test_cases =
    vector<tuple<vector<size_t>, vector<size_t>, vector<size_t>, vector<size_t>, vector<size_t>>>{
        {{0, 1, 2, 3}, {5, 2, 3}, {0, 1, 3, 4}, {5, 3, 4}, {0, 1, 2, 4}},                 // 4D -> 3D -> 4D
        {{5, 2, 3}, {0, 1, 2, 3}, {5, 3, 4}, {0, 1, 3, 4}, {5, 2, 4}},                    // 3D -> 4D -> 3D
        {{0, 1, 2, 3, 4}, {0, 6, 3, 4}, {0, 1, 2, 4, 5}, {0, 6, 4, 5}, {0, 1, 2, 3, 5}},  // 5D -> 4D -> 5D
    };

const auto bea_scalar_modes = vector<size_t>{0, 1, 2, 3};
const auto concat_modes = vector<size_t>{0, 10, 11, 20, 21, 300, 301, 310, 311};
const auto final_add_modes = vector<size_t>{0, 1, 2};

TEST_P(DeReshapeMatMulTest, DeReshapeTests) {}

INSTANTIATE_TEST_SUITE_P(
    TransformationTestsF,
    DeReshapeMatMulTest,
    testing::Combine(testing::ValuesIn(shape_test_cases),  // lhs_idx, rhs_idx, reshape_idx, reshape_idx, reshape_idx
                     testing::ValuesIn(bea_scalar_modes),
                     testing::ValuesIn(concat_modes),
                     testing::ValuesIn(final_add_modes)),
    DeReshapeMatMulTest::getTestCaseName);
