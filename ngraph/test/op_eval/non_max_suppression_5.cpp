//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "ngraph/op/constant.hpp"
#include "ngraph/op/non_max_suppression.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/validation_util.hpp"
#include "runtime/backend.hpp"
#include "util/test_tools.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(op_eval, nonmaxsuppression_center_point_box_format)
{
    std::vector<float> boxes_data = {0.5, 0.5,  1.0, 1.0, 0.5, 0.6,   1.0, 1.0,
                                     0.5, 0.4,  1.0, 1.0, 0.5, 10.5,  1.0, 1.0,
                                     0.5, 10.6, 1.0, 1.0, 0.5, 100.5, 1.0, 1.0};

    std::vector<float> scores_data = {0.9, 0.75, 0.6, 0.95, 0.5, 0.3};

    const int64_t max_output_boxes_per_class_data = 3;
    const float iou_threshold_data = 0.5f;
    const float score_threshold_data = 0.0f;
    const auto box_encoding = op::v5::NonMaxSuppression::BoxEncodingType::CENTER;
    const auto boxes_shape = Shape{1, 6, 4};
    const auto scores_shape = Shape{1, 1, 6};

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);
    auto max_output_boxes_per_class =
        op::Constant::create<int64_t>(element::i64, Shape{}, {max_output_boxes_per_class_data});
    auto iou_threshold = op::Constant::create<float>(element::f32, Shape{}, {iou_threshold_data});
    auto score_threshold =
        op::Constant::create<float>(element::f32, Shape{}, {score_threshold_data});
    auto soft_nms_sigma = op::Constant::create<float>(element::f32, Shape{}, {0.0f});
    auto nms = make_shared<op::v5::NonMaxSuppression>(boxes,
                                                      scores,
                                                      max_output_boxes_per_class,
                                                      iou_threshold,
                                                      score_threshold,
                                                      soft_nms_sigma,
                                                      box_encoding,
                                                      false);

    auto f = make_shared<Function>(nms, ParameterVector{boxes, scores});

    HostTensorVector results(3);
    for (auto& result : results)
    {
        result = make_shared<HostTensor>();
    }
    ASSERT_TRUE(f->evaluate(results,
                           {make_host_tensor<element::Type_t::f32>(boxes_shape, boxes_data),
                            make_host_tensor<element::Type_t::f32>(scores_shape, scores_data)}));

    std::vector<int64_t> expected_selected_indices = {0, 0, 3, 0, 0, 0, 0, 0, 5};
    std::vector<float> expected_selected_scores = {0.0, 0.0, 0.95, 0.0, 0.0, 0.9, 0.0, 0.0, 0.3};
    std::vector<int64_t> expected_valid_outputs = {3};

    EXPECT_EQ(results[0]->get_element_type(), element::i64);
    EXPECT_EQ(results[1]->get_element_type(), element::f32);
    EXPECT_EQ(results[2]->get_element_type(), element::i64);
    EXPECT_EQ(results[0]->get_shape(), (Shape{3, 3}));
    EXPECT_EQ(results[1]->get_shape(), (Shape{3, 3}));
    EXPECT_EQ(results[2]->get_shape(), (Shape{}));
    EXPECT_EQ(read_vector<int64_t>(results[0]), expected_selected_indices);
    EXPECT_EQ(read_vector<float>(results[1]), expected_selected_scores);
    EXPECT_EQ(read_vector<int64_t>(results[2]), expected_valid_outputs);
}

TEST(op_eval, nonmaxsuppression_flipped_coordinates)
{
    std::vector<float> boxes_data = {1.0, 1.0,  0.0, 0.0,  0.0, 0.1,   1.0, 1.1,
                                     0.0, 0.9,  1.0, -0.1, 0.0, 10.0,  1.0, 11.0,
                                     1.0, 10.1, 0.0, 11.1, 1.0, 101.0, 0.0, 100.0};

    std::vector<float> scores_data = {0.9, 0.75, 0.6, 0.95, 0.5, 0.3};

    const int64_t max_output_boxes_per_class_data = 3;
    const float iou_threshold_data = 0.5f;
    const float score_threshold_data = 0.0f;
    const auto box_encoding = op::v5::NonMaxSuppression::BoxEncodingType::CORNER;
    const auto boxes_shape = Shape{1, 6, 4};
    const auto scores_shape = Shape{1, 1, 6};

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);
    auto max_output_boxes_per_class =
        op::Constant::create<int64_t>(element::i64, Shape{}, {max_output_boxes_per_class_data});
    auto iou_threshold = op::Constant::create<float>(element::f32, Shape{}, {iou_threshold_data});
    auto score_threshold =
        op::Constant::create<float>(element::f32, Shape{}, {score_threshold_data});
    auto soft_nms_sigma = op::Constant::create<float>(element::f32, Shape{}, {0.0f});
    auto nms = make_shared<op::v5::NonMaxSuppression>(boxes,
                                                      scores,
                                                      max_output_boxes_per_class,
                                                      iou_threshold,
                                                      score_threshold,
                                                      soft_nms_sigma,
                                                      box_encoding,
                                                      false);

    auto f = make_shared<Function>(nms, ParameterVector{boxes, scores});

    HostTensorVector results(3);
    for (auto& result : results)
    {
        result = make_shared<HostTensor>();
    }
    ASSERT_TRUE(f->evaluate(results,
                            {make_host_tensor<element::Type_t::f32>(boxes_shape, boxes_data),
                             make_host_tensor<element::Type_t::f32>(scores_shape, scores_data)}));

    std::vector<int64_t> expected_selected_indices = {0, 0, 3, 0, 0, 0, 0, 0, 5};
    std::vector<float> expected_selected_scores = {0.0, 0.0, 0.95, 0.0, 0.0, 0.9, 0.0, 0.0, 0.3};
    std::vector<int64_t> expected_valid_outputs = {3};

    EXPECT_EQ(results[0]->get_element_type(), element::i64);
    EXPECT_EQ(results[1]->get_element_type(), element::f32);
    EXPECT_EQ(results[2]->get_element_type(), element::i64);
    EXPECT_EQ(results[0]->get_shape(), (Shape{3, 3}));
    EXPECT_EQ(results[1]->get_shape(), (Shape{3, 3}));
    EXPECT_EQ(results[2]->get_shape(), (Shape{}));
    EXPECT_EQ(read_vector<int64_t>(results[0]), expected_selected_indices);
    EXPECT_EQ(read_vector<float>(results[1]), expected_selected_scores);
    EXPECT_EQ(read_vector<int64_t>(results[2]), expected_valid_outputs);
}

TEST(op_eval, nonmaxsuppression_identical_boxes)
{
    std::vector<float> boxes_data = {0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
                                     1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
                                     0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
                                     1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0};

    std::vector<float> scores_data = {0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9};

    const int64_t max_output_boxes_per_class_data = 3;
    const float iou_threshold_data = 0.5f;
    const float score_threshold_data = 0.0f;
    const auto box_encoding = op::v5::NonMaxSuppression::BoxEncodingType::CORNER;
    const auto boxes_shape = Shape{1, 10, 4};
    const auto scores_shape = Shape{1, 1, 10};

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);
    auto max_output_boxes_per_class =
        op::Constant::create<int64_t>(element::i64, Shape{}, {max_output_boxes_per_class_data});
    auto iou_threshold = op::Constant::create<float>(element::f32, Shape{}, {iou_threshold_data});
    auto score_threshold =
        op::Constant::create<float>(element::f32, Shape{}, {score_threshold_data});
    auto soft_nms_sigma = op::Constant::create<float>(element::f32, Shape{}, {0.0f});
    auto nms = make_shared<op::v5::NonMaxSuppression>(boxes,
                                                      scores,
                                                      max_output_boxes_per_class,
                                                      iou_threshold,
                                                      score_threshold,
                                                      soft_nms_sigma,
                                                      box_encoding,
                                                      false);

    auto f = make_shared<Function>(nms, ParameterVector{boxes, scores});

    HostTensorVector results(3);
    for (auto& result : results)
    {
        result = make_shared<HostTensor>();
    }
    ASSERT_TRUE(f->evaluate(results,
                            {make_host_tensor<element::Type_t::f32>(boxes_shape, boxes_data),
                             make_host_tensor<element::Type_t::f32>(scores_shape, scores_data)}));

    std::vector<int64_t> expected_selected_indices = {0, 0, 0};
    std::vector<float> expected_selected_scores = {0.0, 0.0, 0.9};
    std::vector<int64_t> expected_valid_outputs = {1};

    EXPECT_EQ(results[0]->get_element_type(), element::i64);
    EXPECT_EQ(results[1]->get_element_type(), element::f32);
    EXPECT_EQ(results[2]->get_element_type(), element::i64);
    EXPECT_EQ(results[0]->get_shape(), (Shape{1, 3}));
    EXPECT_EQ(results[1]->get_shape(), (Shape{1, 3}));
    EXPECT_EQ(results[2]->get_shape(), (Shape{}));
    EXPECT_EQ(read_vector<int64_t>(results[0]), expected_selected_indices);
    EXPECT_EQ(read_vector<float>(results[1]), expected_selected_scores);
    EXPECT_EQ(read_vector<int64_t>(results[2]), expected_valid_outputs);
}

TEST(op_eval, nonmaxsuppression_limit_output_size)
{
    std::vector<float> boxes_data = {0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                                     0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                                     0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0};

    std::vector<float> scores_data = {0.9, 0.75, 0.6, 0.95, 0.5, 0.3};

    const int64_t max_output_boxes_per_class_data = 2;
    const float iou_threshold_data = 0.5f;
    const float score_threshold_data = 0.0f;
    const auto box_encoding = op::v5::NonMaxSuppression::BoxEncodingType::CORNER;
    const auto boxes_shape = Shape{1, 6, 4};
    const auto scores_shape = Shape{1, 1, 6};

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);
    auto max_output_boxes_per_class =
        op::Constant::create<int64_t>(element::i64, Shape{}, {max_output_boxes_per_class_data});
    auto iou_threshold = op::Constant::create<float>(element::f32, Shape{}, {iou_threshold_data});
    auto score_threshold =
        op::Constant::create<float>(element::f32, Shape{}, {score_threshold_data});
    auto soft_nms_sigma = op::Constant::create<float>(element::f32, Shape{}, {0.0f});
    auto nms = make_shared<op::v5::NonMaxSuppression>(boxes,
                                                      scores,
                                                      max_output_boxes_per_class,
                                                      iou_threshold,
                                                      score_threshold,
                                                      soft_nms_sigma,
                                                      box_encoding,
                                                      false);

    auto f = make_shared<Function>(nms, ParameterVector{boxes, scores});

    HostTensorVector results(3);
    for (auto& result : results)
    {
        result = make_shared<HostTensor>();
    }
    ASSERT_TRUE(f->evaluate(results,
                            {make_host_tensor<element::Type_t::f32>(boxes_shape, boxes_data),
                             make_host_tensor<element::Type_t::f32>(scores_shape, scores_data)}));

    std::vector<int64_t> expected_selected_indices = {0, 0, 3, 0, 0, 0};
    std::vector<float> expected_selected_scores = {0.0, 0.0, 0.95, 0.0, 0.0, 0.9};
    std::vector<int64_t> expected_valid_outputs = {2};

    EXPECT_EQ(results[0]->get_element_type(), element::i64);
    EXPECT_EQ(results[1]->get_element_type(), element::f32);
    EXPECT_EQ(results[2]->get_element_type(), element::i64);
    EXPECT_EQ(results[0]->get_shape(), (Shape{2, 3}));
    EXPECT_EQ(results[1]->get_shape(), (Shape{2, 3}));
    EXPECT_EQ(results[2]->get_shape(), (Shape{}));
    EXPECT_EQ(read_vector<int64_t>(results[0]), expected_selected_indices);
    EXPECT_EQ(read_vector<float>(results[1]), expected_selected_scores);
    EXPECT_EQ(read_vector<int64_t>(results[2]), expected_valid_outputs);
}

TEST(op_eval, nonmaxsuppression_single_box)
{
    std::vector<float> boxes_data = {0.0, 0.0, 1.0, 1.0};

    std::vector<float> scores_data = {0.9};

    const int64_t max_output_boxes_per_class_data = 3;
    const float iou_threshold_data = 0.5f;
    const float score_threshold_data = 0.0f;
    const auto box_encoding = op::v5::NonMaxSuppression::BoxEncodingType::CORNER;
    const auto boxes_shape = Shape{1, 1, 4};
    const auto scores_shape = Shape{1, 1, 1};

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);
    auto max_output_boxes_per_class =
        op::Constant::create<int64_t>(element::i64, Shape{}, {max_output_boxes_per_class_data});
    auto iou_threshold = op::Constant::create<float>(element::f32, Shape{}, {iou_threshold_data});
    auto score_threshold =
        op::Constant::create<float>(element::f32, Shape{}, {score_threshold_data});
    auto soft_nms_sigma = op::Constant::create<float>(element::f32, Shape{}, {0.0f});
    auto nms = make_shared<op::v5::NonMaxSuppression>(boxes,
                                                      scores,
                                                      max_output_boxes_per_class,
                                                      iou_threshold,
                                                      score_threshold,
                                                      soft_nms_sigma,
                                                      box_encoding,
                                                      false);

    auto f = make_shared<Function>(nms, ParameterVector{boxes, scores});

    HostTensorVector results(3);
    for (auto& result : results)
    {
        result = make_shared<HostTensor>();
    }
    ASSERT_TRUE(f->evaluate(results,
                            {make_host_tensor<element::Type_t::f32>(boxes_shape, boxes_data),
                             make_host_tensor<element::Type_t::f32>(scores_shape, scores_data)}));

    std::vector<int64_t> expected_selected_indices = {0, 0, 0};
    std::vector<float> expected_selected_scores = {0.0, 0.0, 0.9};
    std::vector<int64_t> expected_valid_outputs = {1};

    EXPECT_EQ(results[0]->get_element_type(), element::i64);
    EXPECT_EQ(results[1]->get_element_type(), element::f32);
    EXPECT_EQ(results[2]->get_element_type(), element::i64);
    EXPECT_EQ(results[0]->get_shape(), (Shape{1, 3}));
    EXPECT_EQ(results[1]->get_shape(), (Shape{1, 3}));
    EXPECT_EQ(results[2]->get_shape(), (Shape{}));
    EXPECT_EQ(read_vector<int64_t>(results[0]), expected_selected_indices);
    EXPECT_EQ(read_vector<float>(results[1]), expected_selected_scores);
    EXPECT_EQ(read_vector<int64_t>(results[2]), expected_valid_outputs);
}

TEST(op_eval, nonmaxsuppression_suppress_by_IOU)
{
    std::vector<float> boxes_data = {0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                                     0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                                     0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0};

    std::vector<float> scores_data = {0.9, 0.75, 0.6, 0.95, 0.5, 0.3};

    const int64_t max_output_boxes_per_class_data = 3;
    const float iou_threshold_data = 0.5f;
    const float score_threshold_data = 0.0f;
    const auto box_encoding = op::v5::NonMaxSuppression::BoxEncodingType::CORNER;
    const auto boxes_shape = Shape{1, 6, 4};
    const auto scores_shape = Shape{1, 1, 6};

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);
    auto max_output_boxes_per_class =
        op::Constant::create<int64_t>(element::i64, Shape{}, {max_output_boxes_per_class_data});
    auto iou_threshold = op::Constant::create<float>(element::f32, Shape{}, {iou_threshold_data});
    auto score_threshold =
        op::Constant::create<float>(element::f32, Shape{}, {score_threshold_data});
    auto soft_nms_sigma = op::Constant::create<float>(element::f32, Shape{}, {0.0f});
    auto nms = make_shared<op::v5::NonMaxSuppression>(boxes,
                                                      scores,
                                                      max_output_boxes_per_class,
                                                      iou_threshold,
                                                      score_threshold,
                                                      soft_nms_sigma,
                                                      box_encoding,
                                                      false);

    auto f = make_shared<Function>(nms, ParameterVector{boxes, scores});

    HostTensorVector results(3);
    for (auto& result : results)
    {
        result = make_shared<HostTensor>();
    }
    ASSERT_TRUE(f->evaluate(results,
                            {make_host_tensor<element::Type_t::f32>(boxes_shape, boxes_data),
                             make_host_tensor<element::Type_t::f32>(scores_shape, scores_data)}));

    std::vector<int64_t> expected_selected_indices = {0, 0, 3, 0, 0, 0, 0, 0, 5};
    std::vector<float> expected_selected_scores = {0.0, 0.0, 0.95, 0.0, 0.0, 0.9, 0.0, 0.0, 0.3};
    std::vector<int64_t> expected_valid_outputs = {3};

    EXPECT_EQ(results[0]->get_element_type(), element::i64);
    EXPECT_EQ(results[1]->get_element_type(), element::f32);
    EXPECT_EQ(results[2]->get_element_type(), element::i64);
    EXPECT_EQ(results[0]->get_shape(), (Shape{3, 3}));
    EXPECT_EQ(results[1]->get_shape(), (Shape{3, 3}));
    EXPECT_EQ(results[2]->get_shape(), (Shape{}));
    EXPECT_EQ(read_vector<int64_t>(results[0]), expected_selected_indices);
    EXPECT_EQ(read_vector<float>(results[1]), expected_selected_scores);
    EXPECT_EQ(read_vector<int64_t>(results[2]), expected_valid_outputs);
}

TEST(op_eval, nonmaxsuppression_suppress_by_IOU_and_scores)
{
    std::vector<float> boxes_data = {0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                                     0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                                     0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0};

    std::vector<float> scores_data = {0.9, 0.75, 0.6, 0.95, 0.5, 0.3};

    const int64_t max_output_boxes_per_class_data = 3;
    const float iou_threshold_data = 0.5f;
    const float score_threshold_data = 0.4f;
    const auto box_encoding = op::v5::NonMaxSuppression::BoxEncodingType::CORNER;
    const auto boxes_shape = Shape{1, 6, 4};
    const auto scores_shape = Shape{1, 1, 6};

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);
    auto max_output_boxes_per_class =
        op::Constant::create<int64_t>(element::i64, Shape{}, {max_output_boxes_per_class_data});
    auto iou_threshold = op::Constant::create<float>(element::f32, Shape{}, {iou_threshold_data});
    auto score_threshold =
        op::Constant::create<float>(element::f32, Shape{}, {score_threshold_data});
    auto soft_nms_sigma = op::Constant::create<float>(element::f32, Shape{}, {0.0f});
    auto nms = make_shared<op::v5::NonMaxSuppression>(boxes,
                                                      scores,
                                                      max_output_boxes_per_class,
                                                      iou_threshold,
                                                      score_threshold,
                                                      soft_nms_sigma,
                                                      box_encoding,
                                                      false);

    auto f = make_shared<Function>(nms, ParameterVector{boxes, scores});

    HostTensorVector results(3);
    for (auto& result : results)
    {
        result = make_shared<HostTensor>();
    }
    ASSERT_TRUE(f->evaluate(results,
                            {make_host_tensor<element::Type_t::f32>(boxes_shape, boxes_data),
                             make_host_tensor<element::Type_t::f32>(scores_shape, scores_data)}));

    std::vector<int64_t> expected_selected_indices = {0, 0, 3, 0, 0, 0};
    std::vector<float> expected_selected_scores = {0.0, 0.0, 0.95, 0.0, 0.0, 0.9};
    std::vector<int64_t> expected_valid_outputs = {2};

    EXPECT_EQ(results[0]->get_element_type(), element::i64);
    EXPECT_EQ(results[1]->get_element_type(), element::f32);
    EXPECT_EQ(results[2]->get_element_type(), element::i64);
    EXPECT_EQ(results[0]->get_shape(), (Shape{2, 3}));
    EXPECT_EQ(results[1]->get_shape(), (Shape{2, 3}));
    EXPECT_EQ(results[2]->get_shape(), (Shape{}));
    EXPECT_EQ(read_vector<int64_t>(results[0]), expected_selected_indices);
    EXPECT_EQ(read_vector<float>(results[1]), expected_selected_scores);
    EXPECT_EQ(read_vector<int64_t>(results[2]), expected_valid_outputs);
}

TEST(op_eval, nonmaxsuppression_two_batches)
{
    std::vector<float> boxes_data = {
        0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,   0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
        0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0, 0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
        0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,  0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0};

    std::vector<float> scores_data = {
        0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.9, 0.75, 0.6, 0.95, 0.5, 0.3};

    const int64_t max_output_boxes_per_class_data = 2;
    const float iou_threshold_data = 0.5f;
    const float score_threshold_data = 0.0f;
    const auto box_encoding = op::v5::NonMaxSuppression::BoxEncodingType::CORNER;
    const auto boxes_shape = Shape{2, 6, 4};
    const auto scores_shape = Shape{2, 1, 6};

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);
    auto max_output_boxes_per_class =
        op::Constant::create<int64_t>(element::i64, Shape{}, {max_output_boxes_per_class_data});
    auto iou_threshold = op::Constant::create<float>(element::f32, Shape{}, {iou_threshold_data});
    auto score_threshold =
        op::Constant::create<float>(element::f32, Shape{}, {score_threshold_data});
    auto soft_nms_sigma = op::Constant::create<float>(element::f32, Shape{}, {0.0f});
    auto nms = make_shared<op::v5::NonMaxSuppression>(boxes,
                                                      scores,
                                                      max_output_boxes_per_class,
                                                      iou_threshold,
                                                      score_threshold,
                                                      soft_nms_sigma,
                                                      box_encoding,
                                                      false);

    auto f = make_shared<Function>(nms, ParameterVector{boxes, scores});

    HostTensorVector results(3);
    for (auto& result : results)
    {
        result = make_shared<HostTensor>();
    }
    ASSERT_TRUE(f->evaluate(results,
                            {make_host_tensor<element::Type_t::f32>(boxes_shape, boxes_data),
                             make_host_tensor<element::Type_t::f32>(scores_shape, scores_data)}));

    std::vector<int64_t> expected_selected_indices = {0, 0, 3, 0, 0, 0, 1, 0, 3, 1, 0, 0};
    std::vector<float> expected_selected_scores = {
        0.0, 0.0, 0.95, 0.0, 0.0, 0.9, 1.0, 0.0, 0.95, 1.0, 0.0, 0.9};
    std::vector<int64_t> expected_valid_outputs = {4};

    EXPECT_EQ(results[0]->get_element_type(), element::i64);
    EXPECT_EQ(results[1]->get_element_type(), element::f32);
    EXPECT_EQ(results[2]->get_element_type(), element::i64);
    EXPECT_EQ(results[0]->get_shape(), (Shape{4, 3}));
    EXPECT_EQ(results[1]->get_shape(), (Shape{4, 3}));
    EXPECT_EQ(results[2]->get_shape(), (Shape{}));
    EXPECT_EQ(read_vector<int64_t>(results[0]), expected_selected_indices);
    EXPECT_EQ(read_vector<float>(results[1]), expected_selected_scores);
    EXPECT_EQ(read_vector<int64_t>(results[2]), expected_valid_outputs);
}

TEST(op_eval, nonmaxsuppression_two_classes)
{
    std::vector<float> boxes_data = {0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                                     0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                                     0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0};

    std::vector<float> scores_data = {
        0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.9, 0.75, 0.6, 0.95, 0.5, 0.3};

    const int64_t max_output_boxes_per_class_data = 2;
    const float iou_threshold_data = 0.5f;
    const float score_threshold_data = 0.0f;
    const auto box_encoding = op::v5::NonMaxSuppression::BoxEncodingType::CORNER;
    const auto boxes_shape = Shape{1, 6, 4};
    const auto scores_shape = Shape{1, 2, 6};

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);
    auto max_output_boxes_per_class =
        op::Constant::create<int64_t>(element::i64, Shape{}, {max_output_boxes_per_class_data});
    auto iou_threshold = op::Constant::create<float>(element::f32, Shape{}, {iou_threshold_data});
    auto score_threshold =
        op::Constant::create<float>(element::f32, Shape{}, {score_threshold_data});
    auto soft_nms_sigma = op::Constant::create<float>(element::f32, Shape{}, {0.0f});
    auto nms = make_shared<op::v5::NonMaxSuppression>(boxes,
                                                      scores,
                                                      max_output_boxes_per_class,
                                                      iou_threshold,
                                                      score_threshold,
                                                      soft_nms_sigma,
                                                      box_encoding,
                                                      false);

    auto f = make_shared<Function>(nms, ParameterVector{boxes, scores});

    HostTensorVector results(3);
    for (auto& result : results)
    {
        result = make_shared<HostTensor>();
    }
    ASSERT_TRUE(f->evaluate(results,
                            {make_host_tensor<element::Type_t::f32>(boxes_shape, boxes_data),
                             make_host_tensor<element::Type_t::f32>(scores_shape, scores_data)}));

    std::vector<int64_t> expected_selected_indices = {0, 0, 3, 0, 0, 0, 0, 1, 3, 0, 1, 0};
    std::vector<float> expected_selected_scores = {
        0.0, 0.0, 0.95, 0.0, 0.0, 0.9, 0.0, 1.0, 0.95, 0.0, 1.0, 0.9};
    std::vector<int64_t> expected_valid_outputs = {4};

    EXPECT_EQ(results[0]->get_element_type(), element::i64);
    EXPECT_EQ(results[1]->get_element_type(), element::f32);
    EXPECT_EQ(results[2]->get_element_type(), element::i64);
    EXPECT_EQ(results[0]->get_shape(), (Shape{4, 3}));
    EXPECT_EQ(results[1]->get_shape(), (Shape{4, 3}));
    EXPECT_EQ(results[2]->get_shape(), (Shape{}));
    EXPECT_EQ(read_vector<int64_t>(results[0]), expected_selected_indices);
    EXPECT_EQ(read_vector<float>(results[1]), expected_selected_scores);
    EXPECT_EQ(read_vector<int64_t>(results[2]), expected_valid_outputs);
}
