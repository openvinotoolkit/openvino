// // Copyright (C) 2021 Intel Corporation
// // SPDX-License-Identifier: Apache-2.0
// //

// #include <gtest/gtest.h>

// #include <string>
// #include <memory>
// #include <map>

// #include <ngraph/function.hpp>
// #include <ngraph/opsets/opset5.hpp>
// #include <ngraph/pass/manager.hpp>

// #include <transformations/init_node_info.hpp>
// #include <transformations/preprocessing/std_scale.hpp>
// #include <transformations/preprocessing/mean_image_or_value.hpp>

// #include "common_test_utils/ngraph_test_utils.hpp"


// using namespace testing;
// using namespace ngraph;


// TEST(TransformationTests, Preprocessing_AddStdScale) {
//     std::shared_ptr<Function> f(nullptr), f_ref(nullptr);

//     const Shape data_shape{1, 3, 14, 14};
//     const Shape scale_shape{3, 1, 1};
//     {
//         auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
//         auto relu = std::make_shared<opset5::Relu>(data);
//         f = std::make_shared<Function>(NodeVector{relu}, ParameterVector{data});
//         auto scales = opset5::Constant::create(element::f32, scale_shape,
//             std::vector<float>(shape_size(scale_shape), 2.0f));
//         pass::Manager m;
//         m.register_pass<pass::InitNodeInfo>();
//         m.register_pass<pass::AddStdScale>(pass::AddStdScale::ScaleMap{ { data->get_friendly_name(), scales } });
//         m.run_passes(f);
//     }
//     {
//         auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
//         auto scales = opset5::Constant::create(element::f32, scale_shape,
//             std::vector<float>(shape_size(scale_shape), 2.0f));
//         auto div = std::make_shared<opset5::Divide>(data, scales);
//         auto relu = std::make_shared<opset5::Relu>(div);
//         f_ref = std::make_shared<Function>(NodeVector{relu}, ParameterVector{data});
//     }

//     auto res = compare_functions(f, f_ref);
//     ASSERT_TRUE(res.first) << res.second;
// }

// TEST(TransformationTests, Preprocessing_AddMeanValue) {
//     std::shared_ptr<Function> f(nullptr), f_ref(nullptr);

//     const Shape data_shape{1, 3, 14, 14};
//     const Shape mean_shape{3, 1, 1};
//     {
//         auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
//         auto relu = std::make_shared<opset5::Relu>(data);
//         f = std::make_shared<Function>(NodeVector{relu}, ParameterVector{data});
//         auto meanValues = opset5::Constant::create(element::f32, mean_shape,
//             std::vector<float>(shape_size(mean_shape), 2.0f));
//         pass::Manager m;
//         m.register_pass<pass::InitNodeInfo>();
//         m.register_pass<pass::AddMeanSubtract>(pass::AddMeanSubtract::MeanMap{ { data->get_friendly_name(), meanValues } });
//         m.run_passes(f);
//     }
//     {
//         auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
//         auto meanValues = opset5::Constant::create(element::f32, mean_shape,
//             std::vector<float>(shape_size(mean_shape), 2.0f));
//         auto sub = std::make_shared<opset5::Subtract>(data, meanValues);
//         auto relu = std::make_shared<opset5::Relu>(sub);
//         f_ref = std::make_shared<Function>(NodeVector{relu}, ParameterVector{data});
//     }

//     auto res = compare_functions(f, f_ref);
//     ASSERT_TRUE(res.first) << res.second;
// }

// TEST(TransformationTests, Preprocessing_AddMeanImage) {
//     std::shared_ptr<Function> f(nullptr), f_ref(nullptr);

//     const Shape data_shape{1, 3, 14, 14};
//     const Shape mean_shape{3, 14, 14};
//     {
//         auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
//         auto relu = std::make_shared<opset5::Relu>(data);
//         f = std::make_shared<Function>(NodeVector{relu}, ParameterVector{data});
//         auto meanValues = opset5::Constant::create(element::f32, mean_shape,
//             std::vector<float>(shape_size(mean_shape), 2.0f));
//         pass::Manager m;
//         m.register_pass<pass::InitNodeInfo>();
//         m.register_pass<pass::AddMeanSubtract>(pass::AddMeanSubtract::MeanMap{ { data->get_friendly_name(), meanValues } });
//         m.run_passes(f);
//     }
//     {
//         auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
//         auto meanValues = opset5::Constant::create(element::f32, mean_shape,
//             std::vector<float>(shape_size(mean_shape), 2.0f));
//         auto sub = std::make_shared<opset5::Subtract>(data, meanValues);
//         auto relu = std::make_shared<opset5::Relu>(sub);
//         f_ref = std::make_shared<Function>(NodeVector{relu}, ParameterVector{data});
//     }

//     auto res = compare_functions(f, f_ref);
//     ASSERT_TRUE(res.first) << res.second;
// }

// TEST(TransformationTests, Preprocessing_AddMeanImageAndScale) {
//     std::shared_ptr<Function> f(nullptr), f_ref(nullptr);

//     const Shape data_shape{1, 3, 14, 14};
//     const Shape mean_shape{3, 14, 14};
//     const Shape scale_shape{3, 1, 1};
//     {
//         auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
//         auto relu = std::make_shared<opset5::Relu>(data);
//         f = std::make_shared<Function>(NodeVector{relu}, ParameterVector{data});
//         auto meanValues = opset5::Constant::create(element::f32, mean_shape,
//             std::vector<float>(shape_size(mean_shape), 2.0f));
//         auto scaleValues = opset5::Constant::create(element::f32, scale_shape,
//             std::vector<float>(shape_size(scale_shape), 2.0f));
//         pass::Manager m;
//         m.register_pass<pass::InitNodeInfo>();
//         m.register_pass<pass::AddStdScale>(pass::AddStdScale::ScaleMap{ { data->get_friendly_name(), scaleValues } });
//         m.register_pass<pass::AddMeanSubtract>(pass::AddMeanSubtract::MeanMap{ { data->get_friendly_name(), meanValues } });
//         m.run_passes(f);
//     }
//     {
//         auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
//         auto meanValues = opset5::Constant::create(element::f32, mean_shape,
//             std::vector<float>(shape_size(mean_shape), 2.0f));
//         auto scaleValues = opset5::Constant::create(element::f32, scale_shape,
//             std::vector<float>(shape_size(scale_shape), 2.0f));
//         auto sub = std::make_shared<opset5::Subtract>(data, meanValues);
//         auto div = std::make_shared<opset5::Divide>(sub, scaleValues);
//         auto relu = std::make_shared<opset5::Relu>(div);
//         f_ref = std::make_shared<Function>(NodeVector{relu}, ParameterVector{data});
//     }

//     auto res = compare_functions(f, f_ref);
//     ASSERT_TRUE(res.first) << res.second;
// }

// TEST(TransformationTests, Preprocessing_AddMeanValueAndScale) {
//     std::shared_ptr<Function> f(nullptr), f_ref(nullptr);

//     const Shape data_shape{1, 3, 14, 14};
//     const Shape mean_shape{3, 1, 1};
//     const Shape scale_shape{3, 1, 1};
//     {
//         auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
//         auto relu = std::make_shared<opset5::Relu>(data);
//         f = std::make_shared<Function>(NodeVector{relu}, ParameterVector{data});
//         auto meanValues = opset5::Constant::create(element::f32, mean_shape,
//             std::vector<float>(shape_size(mean_shape), 2.0f));
//         auto scaleValues = opset5::Constant::create(element::f32, scale_shape,
//             std::vector<float>(shape_size(scale_shape), 2.0f));
//         pass::Manager m;
//         m.register_pass<pass::InitNodeInfo>();
//         m.register_pass<pass::AddStdScale>(pass::AddStdScale::ScaleMap{ { data->get_friendly_name(), scaleValues } });
//         m.register_pass<pass::AddMeanSubtract>(pass::AddMeanSubtract::MeanMap{ { data->get_friendly_name(), meanValues } });
//         m.run_passes(f);
//     }
//     {
//         auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
//         auto meanValues = opset5::Constant::create(element::f32, mean_shape,
//             std::vector<float>(shape_size(mean_shape), 2.0f));
//         auto scaleValues = opset5::Constant::create(element::f32, scale_shape,
//             std::vector<float>(shape_size(scale_shape), 2.0f));
//         auto sub = std::make_shared<opset5::Subtract>(data, meanValues);
//         auto div = std::make_shared<opset5::Divide>(sub, meanValues);
//         auto relu = std::make_shared<opset5::Relu>(div);
//         f_ref = std::make_shared<Function>(NodeVector{relu}, ParameterVector{data});
//     }

//     auto res = compare_functions(f, f_ref);
//     ASSERT_TRUE(res.first) << res.second;
// }
