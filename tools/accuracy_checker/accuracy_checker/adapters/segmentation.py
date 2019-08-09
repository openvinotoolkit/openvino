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
import numpy as np
from ..adapters import Adapter
from ..representation import SegmentationPrediction, BrainTumorSegmentationPrediction


class SegmentationAdapter(Adapter):
    __provider__ = 'segmentation'

    def process(self, raw, identifiers=None, frame_meta=None):
        result = []
        frame_meta = frame_meta or [] * len(identifiers)
        raw_outputs = self._extract_predictions(raw, frame_meta)
        for identifier, output in zip(identifiers, raw_outputs[self.output_blob]):
            result.append(SegmentationPrediction(identifier, output))

        return result

    def _extract_predictions(self, outputs_list, meta):
        if not 'tiles_shape' in (meta[-1] or {}):
            return outputs_list[0]
        tiles_shapes = [meta['tiles_shape'] for meta in meta]
        restore_output = []
        offset = 0
        for _, image_tiles_shape in enumerate(tiles_shapes):
            next_offset = offset + image_tiles_shape[0] * image_tiles_shape[1]
            image_tiles = [network_output[self.output_blob] for network_output in outputs_list[offset:next_offset]]
            tiles_columns = image_tiles[::image_tiles_shape[0]]
            image = tiles_columns[0]
            for tile_column in tiles_columns[1:]:
                image = np.concatenate((image, tile_column), axis=3)
            restore_output.append(image.squeeze())
            offset = next_offset

        return {self.output_blob: restore_output}


class BrainTumorSegmentationAdapter(Adapter):
    __provider__ = 'brain_tumor_segmentation'

    def process(self, raw, identifiers=None, frame_meta=None):
        result = []
        frame_meta = frame_meta or [] * len(identifiers)
        raw_outputs = self._extract_predictions(raw, frame_meta)
        for identifier, output in zip(identifiers, raw_outputs[self.output_blob]):
            result.append(BrainTumorSegmentationPrediction(identifier, output))

        return result

    def _extract_predictions(self, outputs_list, meta):
        if not (meta[-1] or {}).get('multi_infer', False):
           return outputs_list[0]

        output_keys = list(outputs_list[0].keys())
        output_map = {}
        for output_key in output_keys:
            output_data = [[output[output_key] for output in outputs_list]]
            output_map[output_key] = output_data

        return output_map
