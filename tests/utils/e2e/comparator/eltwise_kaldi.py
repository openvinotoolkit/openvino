import logging as log
import sys

import numpy as np

from utils.table_utils import make_table
from .provider import ClassProvider


class EltwiseKaldiComparator(ClassProvider):
    __action_name__ = "eltwise_kaldi"
    log.basicConfig(
        format="[ %(levelname)s ] %(message)s",
        level=log.INFO,
        stream=sys.stdout)

    def __init__(self, config, infer_result, reference):
        # Generally, thresholds must be determined manually for each model
        self.a_eps = config.get("a_eps") if config.get("a_eps") else 0.1
        self.r_eps = config.get("r_eps") if config.get("r_eps") else 0.1
        self._config = config
        self.infer_result = infer_result
        self.reference = reference
        self.ignore_results = config.get("ignore_results", False)
        self.target_utterances = config.get("target_utterances", self.infer_result.keys())

    def compare(self):
        log.info("Running Element-Wise comparator with following parameters:\n"
                 "\t\t Average absolute difference threshold: {}\n"
                 "\t\t Average relative difference threshold: {}".format(self.a_eps, self.r_eps))
        if sorted(self.infer_result.keys()) != sorted(self.reference.keys()):
            log.warning("Output layers for comparison doesn't match.\n Output layers in infer results: {}\n"
                        "Output layers in reference: {}".format(sorted(self.infer_result.keys()),
                                                                sorted(self.reference.keys())))
        table_header = [
            "Utterance", "Shape", "Infer range", "Reference range", "RMS err", "RMS rel err"
        ]
        table_rows = []
        utterances = sorted(set(self.infer_result.keys()).intersection(self.target_utterances))
        assert utterances, "No utterances for comparison specified for comparator '{}'".format(
            str(self.__action_name__))
        rms_errors = []
        rel_rms_errors = []
        for utterance in utterances:
            data = self.infer_result[utterance]
            ref = self.reference[utterance]
            if data.shape != ref.shape:
                log.warning("Shape of IE output {} isn't equal with shape of FW output {} for utterance {}. "
                            .format(data.shape, ref.shape, utterance) +
                            "Run Dummy comparator to get statistics.")
                from utils.e2e.comparator.dummy import Dummy
                Dummy({}, infer_result={utterance: data}, reference={utterance: ref}).compare()
            else:
                abs_diff = np.absolute(data - ref)
                rel_diff = np.array(abs_diff / np.maximum(np.absolute(data), np.absolute(ref)))
                # Root mean squared error across current utterance
                utt_rms_err = np.sqrt(np.square(abs_diff).mean())
                utt_rms_rel_err = np.sqrt(np.square(rel_diff).mean())
                rms_errors.append(utt_rms_err)
                rel_rms_errors.append(utt_rms_rel_err)
                infer_max = np.amax(data)
                infer_min = np.amin(data)
                infer_range_str = "[{:.3f}, {:.3f}]".format(infer_min, infer_max)
                ref_max = np.amax(ref)
                ref_min = np.amin(ref)
                ref_range_str = "[{:.3f}, {:.3f}]".format(ref_min, ref_max)
                table_rows.append([
                    utterance, data.shape, infer_range_str, ref_range_str, utt_rms_err, utt_rms_rel_err
                ])
        log.info("Element-Wise comparison statistic:\n{}".format(make_table(table_rows, table_header)))
        # Root mean squared error across all utterances
        total_rms_err = np.sqrt(np.square(rms_errors).mean())
        total_rms_rel_err = np.sqrt(np.square(rel_rms_errors).mean())
        log.info("RMS error across all utterances: {}".format(total_rms_err))
        log.info("RMS rel error across all utterances: {}\n".format(total_rms_rel_err))

        self.status = (total_rms_err < self.a_eps) or (total_rms_rel_err < self.r_eps)
        return self.status
