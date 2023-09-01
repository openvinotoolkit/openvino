import re
import csv
import json
from datetime import datetime
from itertools import product
from tqdm import tqdm
from subprocess import run
from pathlib import Path


class Batch():

    # Dictionary of benchmark parameter names and regex expressions for finding them in logs
    parameters = {"Model Read Time (ms)" : r'(?<=Read model took )[\d.]*',
                        "Model Compile Time (ms)" : r'(?<=Compile model took )[\d.]*',
                        'First Inference Time (ms)': r'(?<=First inference took )[\d.]*',
                        'Iterations Count': r'(?<=Count:\s{12})[\d.]*',
                        'Duration (ms)': r'(?<=Duration:\s{9})[\d.]*',
                        'Latency Median (ms)': r'(?<=Median:\s{8})[\d.]*',
                        'Latency Average (ms)': r'(?<=Average:\s{7})[\d.]*',
                        'Latency Min (ms)': r'(?<=Min:\s{11})[\d.]*',
                        'Latency Max (ms)': r'(?<=Max:\s{11})[\d.]*',
                        'Throughput (fps)': r'(?<=Throughput:\s{3})[\d.]*'}

    def __init__(self, models_folder: str, settings: dict, timestamp: str = "") -> None:
        '''Creates a Batch object and initializes OpenVINO environment and creates file directory'''
        self.home_folder = Path().absolute()
        self.models_folder = Path(models_folder)
        self.models = self.get_models()
        self.output_folder = Path(self.home_folder / "output")
        self.timestamp = timestamp if timestamp else datetime.today().strftime('%m_%d_%Y-%H_%M_%S')
        self.results_folder = Path(self.output_folder / self.timestamp)
        with open(settings, "r") as f:
            self.settings = json.load(f)
        self.configs = list(product(*self.settings.values()))

        # Create output and result folders
        if not self.output_folder.exists():
            self.output_folder.mkdir()
        if not self.results_folder.exists():
            self.results_folder.mkdir()

        # Run OpenVINO environment setup script
        Batch.run_command(f"source {self.home_folder}/setup.sh")

    def get_models(self) -> list:
        '''Return list of valid models in model folder'''
        models = []
        parent_folder = str(self.models_folder) + "/"
        for path in self.models_folder.iterdir():
            # Get subfolders and check if they are quantizations or nested models
            if path.is_dir():
                subfolders = [sub for sub in path.iterdir() if sub.is_dir()]
                if subfolders:
                    # Search for nested models
                    shortened_path = str(path).replace(parent_folder, "")
                    nested_models = [str(sub).replace(parent_folder, "") for sub in subfolders 
                                    if len(re.findall(shortened_path, str(sub))) > 1]
                    if nested_models:
                        models.extend(nested_models)
                    else:
                        models.append(shortened_path)
        return models

    def run_command(command: str) -> str:
        '''Runs a shell command and returns output as a string'''
        result = run(command, shell=True, capture_output=True)
        return result.stdout.decode("utf")

    def benchmark(self, model: str, config: tuple) -> str:
        '''Returns OpenVINO benchmark results for a given model and configuration'''
        hint, quantization, device = config[0], config[1], config[2]
        print("Benchmarking", model, quantization, hint, device)
        path = Path(f"{self.models_folder}/{model}/{quantization}/{model.split('/')[-1]}.xml")
        if not path.is_file():
            return f"Error reading model file for {model}"
        log = Batch.run_command(f"benchmark_app -m {path} -hint {hint} -d {device}")
        return f"Benchmark results for {model} {quantization} {hint} {device}:\n{log}"
    
    def benchmark_all_configs(self, model: str) -> str:
        '''Returns OpenVINO benchmark results for a given model in all configurations'''
        log = ""
        # Iterates through all possible combinations of configuration parameters
        for config in self.configs:
            benchmark = self.benchmark(model, config)
            log += benchmark + "\n" + '-' * 20 + "\n\n"
        # Write benchmark output to text file
        with open(self.results_folder / f"{model.split('/')[-1]}_{self.timestamp}_log.txt", "a") as log_file:
            log_file.writelines(log)
        return log

    def benchmark_all_models(self) -> str:
        '''Returns OpenVINO benchmark results for all models in all configurations'''
        log = ""
        # Iterates through all models in model folder
        for model in tqdm(self.models):
            benchmarks = self.benchmark_all_configs(model)
            log += f"All benchmark results for {model}\n\n" + benchmarks
        return log
    
    def parse_log(self, model: str) -> dict:
        '''
        Parses log file of given model
        Return: Dictionary of results where key = config and value = dictionary of results for that model
        '''
        all_results = {config : {} for config in self.configs}
        with open(self.results_folder / f"{model.split('/')[-1]}_{self.timestamp}_log.txt", "r") as f:
            log = f.read()
            # Iterate through all parameters and assign all regex matches to corresponding config
            for param in Batch.parameters:
                param_results = re.findall(Batch.parameters[param], log)
                for j in range(len(param_results)):
                    all_results[self.configs[j]][param] = param_results[j]
        return all_results
    
    def summarize(self) -> str:
        '''Create a CSV summarizing all log results'''
        summary_filepath = self.results_folder / f"{self.timestamp}_summary.csv"
        with open(summary_filepath, 'w', encoding='UTF8') as summary_csv:
            writer = csv.writer(summary_csv)
            writer.writerow(['Model', 'Hint', 'Quantization', 'Device'] + list(self.parameters.keys()))
            # Iterate through each log file
            logs = list(self.results_folder.iterdir())
            for model in logs:
                if model.is_file() and model.suffix == ".txt":
                    model = str(model).split("/")[-1]
                    model = model[:model.index(f"_{self.timestamp}_log.txt")]
                    log_data = self.parse_log(model)
                    for config in self.configs:
                        # Create row of data for each config and append to CSV file
                        if config in log_data:
                            hint, quantization, device = config[0], config[1], config[2]
                            curr_row = [model, hint, quantization, device]
                            for param in self.parameters:
                                if param in log_data[config]:
                                    curr_row.append(log_data[config][param])
                                else:
                                    curr_row.append("NaN")
                            writer.writerow(curr_row)
        return summary_filepath
