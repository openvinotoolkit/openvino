import shutil
from ..utils.network_info import NetworkInfo


class CalibrationConfiguration:
    """
    Class for parsing input config
    """
    def __init__(
        self,
        config: str,
        precision: str,
        model: str,
        weights: str,
        tmp_directory: str,
        output_model: str,
        output_weights: str,
        cpu_extension: str,
        gpu_extension: str,
        device: str,
        batch_size: int,
        threshold: float,
        ignore_layer_types: list,
        ignore_layer_types_path: str,
        ignore_layer_names: list,
        ignore_layer_names_path: str,
        benchmark_iterations_count: int,
        progress: str,
        threshold_step: float,
        threshold_boundary: float,
        simplified_mode: bool = False
    ):

        self._config = config
        self._precision = precision.upper()
        self._model = model
        self._weights = weights
        self._tmp_directory = tmp_directory
        self._output_model = output_model
        self._output_weights = output_weights
        self._cpu_extension = cpu_extension
        self._gpu_extension = gpu_extension
        self._device = device
        self._batch_size = batch_size
        self._threshold = threshold
        self._ignore_layer_types = ignore_layer_types
        self._ignore_layer_types_path = ignore_layer_types_path
        self._ignore_layer_names = ignore_layer_names
        self._ignore_layer_names_path = ignore_layer_names_path
        self._benchmark_iterations_count = benchmark_iterations_count
        self._progress = progress
        self._threshold_step = threshold_step
        self._threshold_boundary = threshold_boundary
        self._simplified_mode = simplified_mode

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.release()

    def release(self):
        if self.tmp_directory:
            shutil.rmtree(self.tmp_directory)
            self._tmp_directory = None

    @property
    def config(self) -> list:
        return self._config

    @property
    def precision(self) -> str:
        return self._precision

    @property
    def model(self) -> str:
        return self._model

    @property
    def weights(self) -> str:
        return self._weights

    @property
    def tmp_directory(self) -> str:
        return self._tmp_directory

    @property
    def output_model(self) -> str:
        return self._output_model

    @property
    def output_weights(self) -> str:
        return self._output_weights

    @property
    def cpu_extension(self) -> str:
        return self._cpu_extension

    @property
    def gpu_extension(self) -> str:
        return self._gpu_extension

    @property
    def device(self) -> str:
        return self._device

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def threshold(self) -> int:
        return self._threshold

    @property
    def ignore_layer_types(self):
        return self._ignore_layer_types

    @property
    def ignore_layer_types_path(self) -> str:
        return self._ignore_layer_types_path

    @property
    def ignore_layer_names(self):
        return self._ignore_layer_names

    @property
    def ignore_layer_names_path(self) -> str:
        return self._ignore_layer_names_path

    @property
    def benchmark_iterations_count(self) -> int:
        return self._benchmark_iterations_count

    @property
    def progress(self) -> str:
        return self._progress

    @property
    def threshold_step(self) -> float:
        return self._threshold_step

    @property
    def threshold_boundary(self) -> float:
        return self._threshold_boundary

    @property
    def simplified_mode(self) -> bool:
        return self._simplified_mode

class CalibrationConfigurationHelper:
    @staticmethod
    def read_ignore_layer_names(configuration: CalibrationConfiguration):
        ignore_layer_types = configuration.ignore_layer_types

        if configuration.ignore_layer_types_path:
            ignore_layer_types_file = open(configuration.ignore_layer_types_path, 'r')
            ignore_layer_types_from_file = [line.strip() for line in ignore_layer_types_file.readlines()]
            ignore_layer_types.extend(ignore_layer_types_from_file)

        ignore_layer_names = NetworkInfo(configuration.model).get_layer_names(layer_types=ignore_layer_types)

        if configuration.ignore_layer_names_path:
            ignore_layer_names_file = open(configuration.ignore_layer_names_path, 'r')
            ignore_layer_names_from_file = [line.strip() for line in ignore_layer_names_file.readlines()]
            ignore_layer_names.extend(ignore_layer_names_from_file)

        return ignore_layer_names
