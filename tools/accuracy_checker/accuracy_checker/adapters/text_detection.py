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

from collections import defaultdict

import cv2
import numpy as np


from ..adapters import Adapter
from ..config import ConfigValidator, StringField, NumberField, BoolField, ConfigError
from ..representation import TextDetectionPrediction, CharacterRecognitionPrediction


class TextDetectionAdapterConfig(ConfigValidator):
    type = StringField()
    pixel_link_out = StringField()
    pixel_class_out = StringField()


class TextDetectionAdapter(Adapter):
    __provider__ = 'text_detection'

    def validate_config(self):
        text_detection_adapter_config = TextDetectionAdapterConfig('TextDetectionAdapter_Config')
        text_detection_adapter_config.validate(self.launcher_config)

    def configure(self):
        self.pixel_link_out = self.launcher_config['pixel_link_out']
        self.pixel_class_out = self.launcher_config['pixel_class_out']

    def process(self, raw, identifiers=None, frame_meta=None):
        results = []
        predictions = self._extract_predictions(raw, frame_meta)
        raw_output = zip(identifiers, frame_meta, predictions[self.pixel_link_out], predictions[self.pixel_class_out])
        for identifier, current_frame_meta, link_data, cls_data in raw_output:
            link_data = link_data.reshape((1, *link_data.shape))
            cls_data = cls_data.reshape((1, *cls_data.shape))
            link_data_shape = link_data.shape
            new_link_data_shape = (link_data_shape[0], link_data_shape[2], link_data_shape[3], link_data_shape[1] / 2)
            cls_data_shape = cls_data.shape
            new_cls_data_shape = (cls_data_shape[0], cls_data_shape[2], cls_data_shape[3], cls_data_shape[1] / 2)
            link_data = self.softmax(link_data.transpose((0, 2, 3, 1)).reshape(-1))[1::2]
            cls_data = self.softmax(cls_data.transpose((0, 2, 3, 1)).reshape(-1))[1::2]
            mask = self.decode_image_by_join(cls_data, new_cls_data_shape, link_data, new_link_data_shape)
            rects = self.mask_to_boxes(mask, current_frame_meta['image_size'])
            results.append(TextDetectionPrediction(identifier, rects))

        return results

    @staticmethod
    def softmax(data):
        for i in np.arange(start=0, stop=data.size, step=2, dtype=int):
            maximum = max(data[i], data[i + 1])
            data[i] = np.exp(data[i] - maximum)
            data[i + 1] = np.exp(data[i + 1] - maximum)
            sum_data = data[i] + data[i + 1]
            data[i] /= sum_data
            data[i + 1] /= sum_data

        return data

    def decode_image_by_join(self, cls_data, cls_data_shape, link_data, link_data_shape):
        k_cls_conf_threshold = 0.7
        k_link_conf_threshold = 0.7
        height = cls_data_shape[1]
        width = cls_data_shape[2]
        id_pixel_mask = np.argwhere(cls_data >= k_cls_conf_threshold).reshape(-1)
        pixel_mask = cls_data >= k_cls_conf_threshold
        group_mask = {}
        pixel_mask[id_pixel_mask] = True
        points = []
        for i in id_pixel_mask:
            points.append((i % width, i // width))
            group_mask[i] = -1
        link_mask = link_data >= k_link_conf_threshold
        neighbours = link_data_shape[3]
        for point in points:
            neighbour = 0
            point_x, point_y = point
            x_neighbours = [point_x - 1, point_x, point_x + 1]
            y_neighbours = [point_y - 1, point_y, point_y + 1]
            for neighbour_y in y_neighbours:
                for neighbour_x in x_neighbours:
                    if neighbour_x == point_x and neighbour_y == point_y:
                        continue

                    if neighbour_x < 0 or neighbour_x >= width or neighbour_y < 0 or neighbour_y >= height:
                        continue

                    pixel_value = np.uint8(pixel_mask[neighbour_y * width + neighbour_x])
                    link_value = np.uint8(
                        link_mask[int(point_y * width * neighbours + point_x * neighbours + neighbour)]
                    )

                    if pixel_value and link_value:
                        group_mask = self.join(point_x + point_y * width, neighbour_x + neighbour_y * width, group_mask)

                    neighbour += 1

        return self.get_all(points, width, height, group_mask)

    def join(self, point1, point2, group_mask):
        root1 = self.find_root(point1, group_mask)
        root2 = self.find_root(point2, group_mask)
        if root1 != root2:
            group_mask[root1] = root2

        return group_mask

    def get_all(self, points, width, height, group_mask):
        root_map = {}
        mask = np.zeros((height, width))

        for point in points:
            point_x, point_y = point
            point_root = self.find_root(point_x + point_y * width, group_mask)
            if not root_map.get(point_root):
                root_map[point_root] = int(len(root_map) + 1)
            mask[point_y, point_x] = root_map[point_root]

        return mask

    @staticmethod
    def find_root(point, group_mask):
        root = point
        update_parent = False
        while group_mask[root] != -1:
            root = group_mask[root]
            update_parent = True

        if update_parent:
            group_mask[point] = root

        return root

    @staticmethod
    def mask_to_boxes(mask, image_size):
        max_val = np.max(mask).astype(int)
        resized_mask = cv2.resize(
            mask.astype(np.float32), (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST
        )
        bboxes = []
        for i in range(int(max_val + 1)):
            bbox_mask = resized_mask == i
            contours_tuple = cv2.findContours(bbox_mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours_tuple[1] if len(contours_tuple) > 2 else contours_tuple[0]
            if not contours:
                continue
            rect = cv2.minAreaRect(contours[0])
            _, hw, _ = rect
            ignored_height = hw[0] >= image_size[0] - 1
            ignored_width = hw[1] >= image_size[1] - 1
            if ignored_height or ignored_width:
                continue
            box = cv2.boxPoints(rect)
            bboxes.append(box)

        return bboxes


class LPRAdapter(Adapter):
    __provider__ = 'lpr'

    def configure(self):
        if not self.label_map:
            raise ConfigError('LPR adapter requires dataset label map for correct decoding.')

    def process(self, raw, identifiers=None, frame_meta=None):
        raw_output = self._extract_predictions(raw, frame_meta)
        predictions = raw_output[self.output_blob]
        result = []
        for identifier, output in zip(identifiers, predictions):
            decoded_out = self.decode(output.reshape(-1))
            result.append(CharacterRecognitionPrediction(identifier, decoded_out))

        return result

    def decode(self, outputs):
        decode_out = str()
        for output in outputs:
            if output == -1:
                break
            decode_out += str(self.label_map[output])

        return decode_out


class BeamSearchDecoderConfig(ConfigValidator):
    beam_size = NumberField(optional=True, floats=False, min_value=1)
    blank_label = NumberField(optional=True, floats=False, min_value=0)
    softmaxed_probabilities = BoolField(optional=True)


class BeamSearchDecoder(Adapter):
    __provider__ = 'beam_search_decoder'

    def validate_config(self):
        beam_search_decoder_config = BeamSearchDecoderConfig(
            'BeamSearchDecoder_Config',
            BeamSearchDecoderConfig.IGNORE_ON_EXTRA_ARGUMENT
        )
        beam_search_decoder_config.validate(self.launcher_config)

    def configure(self):
        if not self.label_map:
            raise ConfigError('Beam Search Decoder requires dataset label map for correct decoding.')

        self.beam_size = self.launcher_config.get('beam_size', 10)
        self.blank_label = self.launcher_config.get('blank_label', len(self.label_map))
        self.softmaxed_probabilities = self.launcher_config.get('softmaxed_probabilities', False)

    def process(self, raw, identifiers=None, frame_meta=None):
        raw_output = self._extract_predictions(raw, frame_meta)
        output = raw_output[self.output_blob]
        output = np.swapaxes(output, 0, 1)

        result = []
        for identifier, data in zip(identifiers, output):
            if self.softmaxed_probabilities:
                data = np.log(data)
            seq = self.decode(data, self.beam_size, self.blank_label)
            decoded = ''.join(str(self.label_map[char]) for char in seq)
            result.append(CharacterRecognitionPrediction(identifier, decoded))
        return result

    @staticmethod
    def decode(probabilities, beam_size=10, blank_id=None):
        """
         Decode given output probabilities to sequence of labels.
        Arguments:
            probabilities: The output log probabilities for each time step.
            Should be an array of shape (time x output dim).
            beam_size (int): Size of the beam to use during decoding.
            blank_id (int): Index of the CTC blank label.
        Returns the output label sequence.
        """
        def make_new_beam():
            return defaultdict(lambda: (-np.inf, -np.inf))

        def log_sum_exp(*args):
            if all(a == -np.inf for a in args):
                return -np.inf
            a_max = np.max(args)
            lsp = np.log(np.sum(np.exp(a - a_max) for a in args))

            return a_max + lsp

        times, symbols = probabilities.shape
        # Initialize the beam with the empty sequence, a probability of 1 for ending in blank
        # and zero for ending in non-blank (in log space).
        beam = [(tuple(), (0.0, -np.inf))]

        for time in range(times):
            # A default dictionary to store the next step candidates.
            next_beam = make_new_beam()

            for symbol_id in range(symbols):
                current_prob = probabilities[time, symbol_id]

                for prefix, (prob_blank, prob_non_blank) in beam:
                    # If propose a blank the prefix doesn't change.
                    # Only the probability of ending in blank gets updated.
                    if symbol_id == blank_id:
                        next_prob_blank, next_prob_non_blank = next_beam[prefix]
                        next_prob_blank = log_sum_exp(
                            next_prob_blank, prob_blank + current_prob, prob_non_blank + current_prob
                        )
                        next_beam[prefix] = (next_prob_blank, next_prob_non_blank)
                        continue
                    # Extend the prefix by the new character symbol and add it to the beam.
                    # Only the probability of not ending in blank gets updated.
                    end_t = prefix[-1] if prefix else None
                    next_prefix = prefix + (symbol_id,)
                    next_prob_blank, next_prob_non_blank = next_beam[next_prefix]
                    if symbol_id != end_t:
                        next_prob_non_blank = log_sum_exp(
                            next_prob_non_blank, prob_blank + current_prob, prob_non_blank + current_prob
                        )
                    else:
                        # Don't include the previous probability of not ending in blank (prob_non_blank) if symbol
                        #  is repeated at the end. The CTC algorithm merges characters not separated by a blank.
                        next_prob_non_blank = log_sum_exp(next_prob_non_blank, prob_blank + current_prob)

                    next_beam[next_prefix] = (next_prob_blank, next_prob_non_blank)
                    # If symbol is repeated at the end also update the unchanged prefix. This is the merging case.
                    if symbol_id == end_t:
                        next_prob_blank, next_prob_non_blank = next_beam[prefix]
                        next_prob_non_blank = log_sum_exp(next_prob_non_blank, prob_non_blank + current_prob)
                        next_beam[prefix] = (next_prob_blank, next_prob_non_blank)

            beam = sorted(next_beam.items(), key=lambda x: log_sum_exp(*x[1]), reverse=True)[:beam_size]

        best = beam[0]

        return best[0]
