"""Kaldi speech postprocessor."""
from .provider import ClassProvider
import numpy as np
import sys
import logging as log
try:
    from kaldiio import save_ark
except ImportError:
    pass

log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)

class IEScoresToKaldi(ClassProvider):
    """Parser for IE scores"""
    __action_name__ = "parse_ie_scores_to_kaldi_format"

    def __init__(self, config):
        # Save postprocessed data to ark file if path is specified
        self.ark_save_path = config.get("ark_save_path", None)

    def apply(self, data):
        """Parse Inference Engine infer results to get structure comparable with Kaldi scores"""
        restructured_data = {}
        for utterance_name, frames_data in data.items():
            frames = []
            for frame_dim in frames_data:
                for out_name, frame_data in frame_dim.items():
                    frames.extend(frame_data)
            restructured_data.update({utterance_name: np.asarray(frames)})
        if self.ark_save_path:
            log.info("Saving Inference Engine output to {}".format(self.ark_save_path))
            save_ark(self.ark_save_path, restructured_data)
        return restructured_data
