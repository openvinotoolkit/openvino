"""
Copyright (c) 2019 Intel Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import cv2
import numpy as np
import pytest

from accuracy_checker.config import ConfigError
from accuracy_checker.preprocessor import (
    Crop,
    Normalize,
    Preprocessor,
    Resize,
    Flip,
    BgrToRgb,
    CropRect,
    ExtendAroundRect,
    PointAligner
)
from accuracy_checker.preprocessor.preprocessing_executor import PreprocessingExecutor
from accuracy_checker.dataset import DataRepresentation


class TestResize:
    def test_default_resize(self, mocker):
        cv2_resize_mock = mocker.patch('accuracy_checker.preprocessor.preprocessors.cv2.resize')
        resize = Preprocessor.provide('resize', {'type': 'resize', 'size': 200})

        input_mock = mocker.Mock()
        resize(DataRepresentation(input_mock))

        assert not resize.use_pil
        assert resize.dst_width == 200
        assert resize.dst_height == 200
        cv2_resize_mock.assert_called_once_with(
            input_mock, (200, 200), interpolation=Resize.OPENCV_INTERPOLATION['LINEAR']
        )

    def test_custom_resize(self, mocker):
        cv2_resize_mock = mocker.patch('accuracy_checker.preprocessor.preprocessors.cv2.resize')

        resize = Preprocessor.provide(
            'resize', {'type': 'resize', 'dst_width': 126, 'dst_height': 128, 'interpolation': 'CUBIC'}
        )

        input_mock = mocker.Mock()
        resize(DataRepresentation(input_mock))

        assert not resize.use_pil
        assert resize.dst_width == 126
        assert resize.dst_height == 128
        cv2_resize_mock.assert_called_once_with(
            input_mock, (126, 128),
            interpolation=Resize.OPENCV_INTERPOLATION['CUBIC']
        )

    def test_resize_without_save_aspect_ratio(self):
        name = 'mock_preprocessor'
        config = {'type': 'resize', 'dst_width': 150, 'dst_height': 150}
        input_image = np.ones((100, 50, 3))
        resize = Preprocessor.provide('resize', config, name)

        result = resize(DataRepresentation(input_image)).data

        assert result.shape == (150, 150, 3)

    def test_resize_save_aspect_ratio_unknown_raise_config_error(self):
        with pytest.raises(ConfigError):
            Preprocessor.provide(
                'resize', {'type': 'resize', 'dst_width': 100, 'dst_height': 150, 'aspect_ratio_scale': 'unknown'}
            )

    def test_resize_save_aspect_ratio_height(self):
        input_image = np.ones((100, 50, 3))
        resize = Preprocessor.provide('resize', {
            'type': 'resize', 'dst_width': 100, 'dst_height': 150,
            'interpolation': 'CUBIC', 'aspect_ratio_scale': 'height'
        })
        result = resize(DataRepresentation(input_image)).data

        assert result.shape == (300, 100, 3)

    def test_resize_save_aspect_ratio_width(self):
        input_image = np.ones((100, 50, 3))
        resize = Preprocessor.provide('resize', {
            'type': 'resize', 'dst_width': 150, 'dst_height': 150, 'aspect_ratio_scale': 'width'
        })
        result = resize(DataRepresentation(input_image)).data

        assert result.shape == (150, 75, 3)

    def test_resize_save_aspect_ratio_for_greater_dim(self):
        input_image = np.ones((100, 50, 3))
        resize = Preprocessor.provide('resize', {
            'type': 'resize',
            'dst_width': 100,
            'dst_height': 150,
            'aspect_ratio_scale': 'greater'
        })
        result = resize(DataRepresentation(input_image)).data

        assert result.shape == (300, 100, 3)

    def test_resize_to_negative_size_raise_config_error(self):
        with pytest.raises(ConfigError):
            Preprocessor.provide('resize', {'type': 'resize', 'size': -100})

    def test_resize_to_negative_destination_width_raise_config_error(self):
        with pytest.raises(ConfigError):
            Preprocessor.provide('resize', {'type': 'resize', 'dst_width': -100, 'dst_height': 100})

    def test_resize_to_negative_destination_height_raise_config_error(self):
        with pytest.raises(ConfigError):
            Preprocessor.provide('resize', {'type': 'resize', 'dst_width': 100, 'dst_height': -100})

    def test_resize_with_both_provided_size_and_dst_height_dst_width_warn(self):
        input_image = np.ones((100, 50, 3))

        with pytest.warns(None) as warnings:
            resize = Preprocessor.provide(
                'resize', {'type': 'resize', 'dst_width': 100, 'dst_height': 100, 'size': 200}
            )
            assert len(warnings) == 1
            result = resize(DataRepresentation(input_image)).data
            assert result.shape == (200, 200, 3)

    def test_resize_provided_only_dst_height_raise_config_error(self):
        with pytest.raises(ValueError):
            Preprocessor.provide('resize', {'type': 'resize', 'dst_height': 100})

    def test_resize_provided_only_dst_width_raise_config_error(self):
        with pytest.raises(ValueError):
            Preprocessor.provide('resize', {'type': 'resize', 'dst_width': 100})


class TestNormalization:
    def test_normalization_without_mean_and_std_raise_config_error(self):
        with pytest.raises(ConfigError):
            Preprocessor.provide('normalization', {'type': 'normalization'})

    def test_custom_normalization_with_mean(self):
        normalization = Preprocessor.provide('normalization', {'type': 'normalization', 'mean': '(1, 2, 3)'})
        source = np.full_like((3, 300, 300), 100)
        input_ref = source.copy() - (1, 2, 3)
        result = normalization(DataRepresentation(source))

        assert normalization.mean == (1, 2, 3)
        assert normalization.std is None
        assert np.all(input_ref == result.data)
        assert result.metadata == {'image_size': (3,)}

    def test_custom_normalization_with_precomputed_mean(self):
        normalization = Preprocessor.provide('normalization', {'type': 'normalization', 'mean': 'cifar10'})

        source = np.full_like((3, 300, 300), 100)
        input_ref = source.copy() - normalization.PRECOMPUTED_MEANS['cifar10']
        result = normalization(DataRepresentation(source))

        assert normalization.mean == normalization.PRECOMPUTED_MEANS['cifar10']
        assert normalization.std is None
        assert np.all(input_ref == result.data)
        assert result.metadata == {'image_size': (3,)}

    def test_custom_normalization_with_mean_as_scalar(self):
        normalization = Preprocessor.provide('normalization', {'type': 'normalization', 'mean': '1'})

        source = np.full_like((3, 300, 300), 100)
        input_ref = source.copy() - 1
        result = normalization(DataRepresentation(source))

        assert normalization.mean == (1.0, )
        assert normalization.std is None
        assert np.all(input_ref == result.data)
        assert result.metadata == {'image_size': (3,)}

    def test_custom_normalization_with_std(self):
        normalization = Preprocessor.provide('normalization', {'type': 'normalization', 'std': '(1, 2, 3)'})

        source = np.full_like((3, 300, 300), 100)
        input_ref = source.copy() / (1, 2, 3)
        result = normalization(DataRepresentation(source))

        assert normalization.mean is None
        assert normalization.std == (1, 2, 3)
        assert np.all(input_ref == result.data)
        assert result.metadata == {'image_size': (3,)}

    def test_custom_normalization_with_precomputed_std(self):
        normalization = Preprocessor.provide('normalization', {'type': 'normalization', 'std': 'cifar10'})

        source = np.full_like((3, 300, 300), 100)
        input_ref = source.copy() / normalization.PRECOMPUTED_STDS['cifar10']
        result = normalization(DataRepresentation(source))

        assert normalization.mean is None
        assert normalization.std == normalization.PRECOMPUTED_STDS['cifar10']
        assert np.all(input_ref == result.data)
        assert result.metadata == {'image_size': (3,)}

    def test_custom_normalization_with_std_as_scalar(self):
        normalization = Preprocessor.provide('normalization', {'type': 'normalization', 'std': '2'})
        source = np.full_like((3, 300, 300), 100)
        input_ref = source.copy() / 2
        result = normalization(DataRepresentation(source))

        assert normalization.mean is None
        assert normalization.std == (2.0, )
        assert np.all(input_ref == result.data)
        assert result.metadata == {'image_size': (3,)}

    def test_custom_normalization_with_mean_and_std(self):
        normalization = Preprocessor.provide(
            'normalization', {'type': 'normalization', 'mean': '(1, 2, 3)', 'std': '(4, 5, 6)'}
        )

        input_ = np.full_like((3, 300, 300), 100)
        input_ref = (input_ - (1, 2, 3)) / (4, 5, 6)
        result = normalization(DataRepresentation(input_))

        assert normalization.mean == (1, 2, 3)
        assert normalization.std == (4, 5, 6)
        assert np.all(input_ref == result.data)
        assert result.metadata == {'image_size': (3,)}

    def test_custom_normalization_with_mean_and_std_as_scalars(self):
        normalization = Preprocessor.provide('normalization', {'type': 'normalization', 'mean': '2', 'std': '5'})

        input_ = np.full_like((3, 300, 300), 100)
        input_ref = (input_ - (2, )) / (5, )
        result = normalization(DataRepresentation(input_))

        assert normalization.mean == (2, )
        assert normalization.std == (5, )
        assert np.all(input_ref == result.data)
        assert result.metadata == {'image_size': (3,)}

    def test_normalization_with_zero_in_std_values_raise_config_error(self):
        with pytest.raises(ConfigError):
            Preprocessor.provide('normalization', {'type': 'normalization', 'std': '(4, 0, 6)'})

    def test_normalization_with_zero_as_std_value_raise_config_error(self):
        with pytest.raises(ConfigError):
            Preprocessor.provide('normalization', {'type': 'normalization', 'std': '0'})

    def test_normalization_with_not_channel_wise_mean_list_raise_config_error(self):
        with pytest.raises(ConfigError):
            Preprocessor.provide('normalization', {'type': 'normalization', 'mean': '3, 2'})

    def test_normalization_with_not_channel_wise_std_list_raise_config_error(self):
        with pytest.raises(ConfigError):
            Preprocessor.provide('normalization', {'type': 'normalization', 'std': '3, 2'})

    def test_normalization_with_unknown_precomputed_mean_raise_config_error(self):
        with pytest.raises(ValueError):
            Preprocessor.provide('normalization', {'type': 'normalization', 'mean': 'unknown'})

    def test_normalization_with_unknown_precomputed_std_raise_config_error(self):
        with pytest.raises(ValueError):
            Preprocessor.provide('normalization', {'type': 'normalization', 'std': 'unknown'})


class TestPreprocessingEvaluator:
    def test_preprocessing_evaluator(self):
        config = [{'type': 'normalization', 'mean': '(1, 2, 3)'}, {'type': 'resize', 'size': 200}]
        preprocessor = PreprocessingExecutor(config)

        assert 2 == len(preprocessor.processors)
        assert isinstance(preprocessor.processors[0], Normalize)
        assert isinstance(preprocessor.processors[1], Resize)
        assert preprocessor.processors[0].mean == (1, 2, 3)
        assert preprocessor.processors[1].dst_width == 200


class TestCrop:
    def test_crop_higher(self):
        crop = Crop({'dst_width': 50, 'dst_height': 33, 'type': 'crop'})
        image = np.zeros((100, 100, 3))
        image_rep = crop(DataRepresentation(image))

        assert image_rep.data.shape == (33, 50, 3)
        assert image_rep.metadata == {'image_size': (100, 100, 3)}

    def test_crop_to_size(self):
        crop = Crop({'size': 50, 'type': 'crop'})
        image = np.zeros((100, 100, 3))
        image_rep = crop(DataRepresentation(image))

        assert image_rep.data.shape == (50, 50, 3)
        assert image_rep.metadata == {'image_size': (100, 100, 3)}

    def test_crop_higher_non_symmetric(self):
        crop = Crop({'dst_width': 50, 'dst_height': 12, 'type': 'crop'})
        image = np.zeros((70, 50, 3))
        image_rep = crop(DataRepresentation(image))

        assert image_rep.data.shape == (12, 50, 3)
        assert image_rep.metadata == {'image_size': (70, 50, 3)}

    def test_crop_less(self):
        crop = Crop({'dst_width': 151, 'dst_height': 42, 'type': 'crop'})
        image = np.zeros((30, 30, 3))
        image_rep = crop(DataRepresentation(image))

        assert image_rep.data.shape == (42, 151, 3)
        assert image_rep.metadata == {'image_size': (30, 30, 3)}

    def test_crop_less_non_symmetric(self):
        crop = Crop({'dst_width': 42, 'dst_height': 151, 'type': 'crop'})
        image = np.zeros((30, 40, 3))
        image_rep = crop(DataRepresentation(image))

        assert image_rep.data.shape == (151, 42, 3)
        assert image_rep.metadata == {'image_size': (30, 40, 3)}

    def test_crop_to_negative_size_raise_config_error(self):
        with pytest.raises(ConfigError):
            Crop({'size': -151, 'type': 'crop'})

    def test_crop_to_negative_destination_width_raise_config_error(self):
        with pytest.raises(ConfigError):
            Crop({'dst_width': -100, 'dst_height': 100, 'type': 'crop'})

    def test_crop_to_negative_destination_height_raise_config_error(self):
        with pytest.raises(ConfigError):
            Crop({'dst_width': 100, 'dst_height': -100, 'type': 'crop'})

    def test_crop_with_both_provided_size_and_dst_height_dst_width_warn(self):
        image = np.zeros((30, 40, 3))
        with pytest.warns(None) as warnings:
            crop = Crop({'dst_width': 100, 'dst_height': 100, 'size': 200, 'type': 'crop'})
            assert len(warnings) == 1
            result = crop.process(DataRepresentation(image))
            assert result.data.shape == (200, 200, 3)
            assert result.metadata == {'image_size': (30, 40, 3)}


class TestFlip:
    def test_horizontal_flip(self):
        image = np.random.randint(0, 255, (30, 40, 3))
        expected_image = cv2.flip(image, 0)
        flip = Flip({'type': 'flip', 'mode': 'horizontal'})
        assert np.array_equal(expected_image, flip.process(DataRepresentation(image)).data)

    def test_vertical_flip(self):
        image = np.random.randint(0, 255, (30, 40, 3))
        expected_image = cv2.flip(image, 1)
        flip = Flip({'type': 'flip', 'mode': 'vertical'})
        assert np.array_equal(expected_image, flip.process(DataRepresentation(image)).data)

    def test_flip_raise_config_error_if_mode_not_provided(self):
        with pytest.raises(ConfigError):
            Flip({'type': 'flip'})

    def test_flip_raise_config_error_if_mode_unknown(self):
        with pytest.raises(ConfigError):
            Flip({'type': 'flip', 'mode': 'unknown'})


class TestBGRtoRGB:
    def test_bgr_to_rgb(self):
        image = np.random.randint(0, 255, (30, 40, 3)).astype(np.uint8)
        expected_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bgr_to_rgb = BgrToRgb({'type': 'bgr_to_rgb'})
        assert np.array_equal(expected_image, bgr_to_rgb.process(DataRepresentation(image)).data)


class TestCropRect:
    def test_crop_rect_if_rect_not_provided(self):
        image = np.zeros((30, 40, 3))
        crop_rect = CropRect({'type': 'crop_rect'})
        assert np.array_equal(image, crop_rect(image, {}))

    def test_crop_rect_if_rect_equal_image(self):
        image = np.zeros((30, 40, 3))
        crop_rect = CropRect({'type': 'crop_rect'})
        assert np.array_equal(image, crop_rect(DataRepresentation(image), {'rect': [0, 0, 40, 30]}).data)

    def test_crop_rect(self):
        image = np.zeros((30, 40, 3))
        image[:, 20:, :] = 1
        expected_image = np.ones((30, 20, 3))
        crop_rect = CropRect({'type': 'crop_rect'})
        assert np.array_equal(expected_image, crop_rect(DataRepresentation(image), {'rect': [20, 0, 40, 30]}).data)

    def test_crop_rect_negative_coordinates_of_rect(self):
        image = np.zeros((30, 40, 3))
        image[:, 20:, :] = 1
        expected_image = image
        crop_rect = CropRect({'type': 'crop_rect'})
        assert np.array_equal(expected_image, crop_rect(DataRepresentation(image), {'rect': [-20, 0, 40, 30]}).data)

    def test_crop_rect_more_image_size_coordinates_of_rect(self):
        image = np.zeros((30, 40, 3))
        image[:, 20:, :] = 1
        expected_image = np.ones((30, 20, 3))
        crop_rect = CropRect({'type': 'crop_rect'})
        assert np.array_equal(expected_image, crop_rect(DataRepresentation(image), {'rect': [20, 0, 40, 50]}).data)


class TestExtendAroundRect:
    def test_default_extend_around_rect_without_rect(self):
        image = np.random.randint(0, 255, (30, 40, 3)).astype(np.uint8)
        expected_image = image
        extend_image_around_rect = ExtendAroundRect({'type': 'extend_around_rect'})
        assert np.array_equal(expected_image, extend_image_around_rect(DataRepresentation(image), {}).data)

    def test_default_extend_around_rect(self):
        image = np.random.randint(0, 255, (30, 40, 3)).astype(np.uint8)
        expected_image = image
        extend_image_around_rect = ExtendAroundRect({'type': 'extend_around_rect'})
        assert np.array_equal(
            expected_image, extend_image_around_rect(DataRepresentation(image), {'rect': [20, 0, 40, 30]}).data
        )

    def test_extend_around_rect_with_positive_augmentation(self):
        image = np.random.randint(0, 255, (30, 40, 3)).astype(np.uint8)
        expected_image = cv2.copyMakeBorder(image, int(15.5), int(31), int(0), int(11), cv2.BORDER_REPLICATE)
        extend_image_around_rect = ExtendAroundRect({'type': 'extend_around_rect', 'augmentation_param': 0.5})
        assert np.array_equal(
            expected_image, extend_image_around_rect(DataRepresentation(image), {'rect': [20, 0, 40, 30]}).data
        )

    def test_extend_around_rect_with_negative_augmentation(self):
        image = np.random.randint(0, 255, (30, 40, 3)).astype(np.uint8)
        expected_image = image
        extend_image_around_rect = ExtendAroundRect({'type': 'extend_around_rect', 'augmentation_param': -0.5})
        assert np.array_equal(
            expected_image, extend_image_around_rect(DataRepresentation(image), {'rect': [20, 0, 40, 30]}).data
        )

    def test_extend_around_rect_with_rect_equal_image(self):
        image = np.random.randint(0, 255, (30, 40, 3)).astype(np.uint8)
        expected_image = cv2.copyMakeBorder(image, int(15.5), int(31), int(20.5), int(41), cv2.BORDER_REPLICATE)
        extend_image_around_rect = ExtendAroundRect({'type': 'extend_around_rect', 'augmentation_param': 0.5})
        assert np.array_equal(
            expected_image, extend_image_around_rect(DataRepresentation(image), {'rect': [0, 0, 40, 30]}).data
        )

    def test_extend_around_rect_negative_coordinates_of_rect(self):
        image = np.random.randint(0, 255, (30, 40, 3)).astype(np.uint8)
        expected_image = cv2.copyMakeBorder(image, int(15.5), int(31), int(20.5), int(41), cv2.BORDER_REPLICATE)
        extend_image_around_rect = ExtendAroundRect({'type': 'extend_around_rect', 'augmentation_param': 0.5})
        assert np.array_equal(
            expected_image, extend_image_around_rect(DataRepresentation(image), {'rect': [-20, 0, 40, 30]}).data
        )

    def test_extend_around_rect_more_image_size_coordinates_of_rect(self):
        image = np.random.randint(0, 255, (30, 40, 3)).astype(np.uint8)
        expected_image = cv2.copyMakeBorder(image, int(15.5), int(31), int(0), int(11), cv2.BORDER_REPLICATE)
        extend_image_around_rect = ExtendAroundRect({'type': 'extend_around_rect', 'augmentation_param': 0.5})
        assert np.array_equal(
            expected_image, extend_image_around_rect(DataRepresentation(image), {'rect': [20, 0, 40, 50]}).data
        )


class TestPointAlignment:
    def test_point_alignment_width_negative_size_raise_config_error(self):
        with pytest.raises(ConfigError):
            PointAligner({'type': 'point_alignment', 'size': -100})

    def test_point_alignment_negative_destination_width_raise_config_error(self):
        with pytest.raises(ConfigError):
            PointAligner({'type': 'point_alignment', 'dst_width': -100, 'dst_height': 100})

    def test_point_alignment_to_negative_destination_height_raise_config_error(self):
        with pytest.raises(ValueError):
            PointAligner({'type': 'point_alignment', 'dst_width': 100, 'dst_height': -100})

    def test_point_alignment_provided_only_dst_height_raise_config_error(self):
        with pytest.raises(ValueError):
            PointAligner({'type': 'point_alignment', 'dst_height': 100})

    def test_point_alignment_provided_only_dst_width_raise_config_error(self):
        with pytest.raises(ValueError):
            PointAligner({'type': 'point_alignment', 'dst_width': 100})

    def test_point_alignment_both_provided_size_and_dst_height_dst_width_warn(self):
        input_image = np.ones((100, 50, 3))

        with pytest.warns(None) as warnings:
            point_aligner = PointAligner({'type': 'point_alignment', 'dst_width': 100, 'dst_height': 100, 'size': 200})
            assert len(warnings) == 1
            result = point_aligner(DataRepresentation(input_image), {}).data
            assert result.shape == (100, 50, 3)

    def test_point_alignment_not_provided_points_im_meta(self):
        input_image = np.ones((100, 50, 3))

        point_aligner = PointAligner({'type': 'point_alignment', 'dst_width': 100, 'dst_height': 100})
        result = point_aligner(DataRepresentation(input_image), {}).data
        assert result.shape == (100, 50, 3)

    def test_point_alignment_default_use_normalization(self):
        image = np.random.randint(0, 255, (40, 40, 3)).astype(np.uint8)

        point_aligner = PointAligner({'type': 'point_alignment', 'dst_width': 40, 'dst_height': 40})
        result = point_aligner(
            DataRepresentation(image), {'keypoints': PointAligner.ref_landmarks.reshape(-1).tolist()}
        ).data
        transformation_matrix = point_aligner.transformation_from_points(
            point_aligner.ref_landmarks * 40, point_aligner.ref_landmarks
        )
        expected_result = cv2.warpAffine(image, transformation_matrix, (40, 40), flags=cv2.WARP_INVERSE_MAP)

        assert np.array_equal(result, expected_result)

    def test_point_alignment_use_normalization(self):
        image = np.random.randint(0, 255, (40, 40, 3)).astype(np.uint8)

        point_aligner = PointAligner({'type': 'point_alignment', 'dst_width': 40, 'dst_height': 40, 'normalize': True})
        result = point_aligner(
            DataRepresentation(image), {'keypoints': PointAligner.ref_landmarks.reshape(-1).tolist()}
        ).data
        transformation_matrix = point_aligner.transformation_from_points(
            point_aligner.ref_landmarks * 40, point_aligner.ref_landmarks
        )
        expected_result = cv2.warpAffine(image, transformation_matrix, (40, 40), flags=cv2.WARP_INVERSE_MAP)

        assert np.array_equal(result, expected_result)

    def test_point_alignment_without_normalization(self):
        image = np.random.randint(0, 255, (40, 40, 3)).astype(np.uint8)

        point_aligner = PointAligner({'type': 'point_alignment', 'dst_width': 40, 'dst_height': 40, 'normalize': False})
        result = point_aligner(
            DataRepresentation(image), {'keypoints': PointAligner.ref_landmarks.reshape(-1).tolist()}
        ).data
        transformation_matrix = point_aligner.transformation_from_points(
            point_aligner.ref_landmarks * 40, point_aligner.ref_landmarks * 40
        )
        expected_result = cv2.warpAffine(image, transformation_matrix, (40, 40), flags=cv2.WARP_INVERSE_MAP)

        assert np.array_equal(result, expected_result)

    def test_point_alignment_with_drawing_points(self):
        image = np.random.randint(0, 255, (40, 40, 3)).astype(np.uint8)

        point_aligner = PointAligner({
            'type': 'point_alignment', 'dst_width': 40, 'dst_height': 40, 'draw_points': True
        })
        result = point_aligner(
            DataRepresentation(image), {'keypoints': PointAligner.ref_landmarks.reshape(-1).tolist()}
        ).data
        transformation_matrix = point_aligner.transformation_from_points(
            point_aligner.ref_landmarks * 40, point_aligner.ref_landmarks
        )
        expected_result = image
        for point in PointAligner.ref_landmarks:
            cv2.circle(expected_result, (int(point[0]), int(point[1])), 5, (255, 0, 0), -1)
        expected_result = cv2.warpAffine(expected_result, transformation_matrix, (40, 40), flags=cv2.WARP_INVERSE_MAP)

        assert np.array_equal(result, expected_result)

    def test_point_alignment_with_resizing(self):
        image = np.random.randint(0, 255, (80, 80, 3)).astype(np.uint8)

        point_aligner = PointAligner({'type': 'point_alignment', 'size': 40})
        result = point_aligner(
            DataRepresentation(image), {'keypoints': PointAligner.ref_landmarks.reshape(-1).tolist()}
        ).data
        transformation_matrix = point_aligner.transformation_from_points(
            point_aligner.ref_landmarks * 40, point_aligner.ref_landmarks * 0.5
        )
        expected_result = cv2.resize(image, (40, 40))
        expected_result = cv2.warpAffine(expected_result, transformation_matrix, (40, 40), flags=cv2.WARP_INVERSE_MAP)

        assert np.array_equal(result, expected_result)


class TestPreprocessorExtraArgs:
    def test_resize_raise_config_error_on_extra_args(self):
        with pytest.raises(ConfigError):
            Preprocessor.provide('resize', {'type': 'resize', 'size': 1, 'something_extra': 'extra'})

    def test_normalization_raise_config_error_on_extra_args(self):
        with pytest.raises(ConfigError):
            Preprocessor.provide('normalization', {'type': 'normalization', 'mean': 0, 'something_extra': 'extra'})

    def test_bgr_to_rgb_raise_config_error_on_extra_args(self):
        with pytest.raises(ConfigError):
            Preprocessor.provide('bgr_to_rgb',  {'type': 'bgr_to_rgb', 'something_extra': 'extra'})

    def test_flip_raise_config_error_on_extra_args(self):
        with pytest.raises(ConfigError):
            Preprocessor.provide('flip', {'type': 'flip', 'something_extra': 'extra'})

    def test_crop_accuracy_raise_config_error_on_extra_args(self):
        with pytest.raises(ConfigError):
            Preprocessor.provide('crop', {'type': 'crop', 'size': 1, 'something_extra': 'extra'})

    def test_extend_around_rect_raise_config_error_on_extra_args(self):
        with pytest.raises(ConfigError):
            Preprocessor.provide('extend_around_rect', {'type': 'extend_around_rect', 'something_extra': 'extra'})

    def test_point_alignment_raise_config_error_on_extra_args(self):
        with pytest.raises(ConfigError):
            Preprocessor.provide('point_alignment', {'type': 'point_alignment', 'something_extra': 'extra'})
