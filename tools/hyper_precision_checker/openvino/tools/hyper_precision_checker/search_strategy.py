from openvino.runtime import InferRequest
import copy
from .logging import logger


can_run_bf16_ops = {"Convolution", "FullyConnected", "RNNCell", "RNNSeq",
                    "MatMul", "MulAdd", "Add", "ROIPooling", "Interpolate",
                    "MVN", "MaxPool"}


class FP32FallbackSearcher:
    def __init__(self, total_size: int, latency: float) -> None:
        self.force_use_fp32_ops = set()
        self.bf16_ops_list = []
        self.potential_bf16_ops_list = []
        self.best_config = [['all', total_size, latency]]
        self.current_try_op = None

    def init_searcher(self, namelist):
        self.bf16_ops_list = []
        for it in namelist:
            if it[1] in can_run_bf16_ops:
                self.bf16_ops_list.append(it[0])
        self.potential_bf16_ops_list = self.bf16_ops_list.copy()

    def has_next(self):
        if len(self.potential_bf16_ops_list) > 0:
            return True
        return False

    def next_config(self):
        # logger.debug(f"%%%  best_config={best_config}   {len(potential_bf16_ops_list)} round to search %%%")
        if len(self.potential_bf16_ops_list) == 0:
            return None
        potential_force_fp32_set = self.force_use_fp32_ops.copy()
        self.current_try_op = self.potential_bf16_ops_list.pop(0)
        logger.info(f"try to fallback {self.current_try_op} to FP32")
        potential_force_fp32_set.add(self.current_try_op)
        logger.debug(f"{potential_force_fp32_set} will fallback to FP32")
        return potential_force_fp32_set

    def store_config(self, right: int, average_time: float):
        if self.current_try_op is not None:
            self.force_use_fp32_ops.add(self.current_try_op)
            self.best_config.append(
                [self.force_use_fp32_ops.copy(), right, average_time])
        else:
            self.best_config.append(["None", right, average_time])
