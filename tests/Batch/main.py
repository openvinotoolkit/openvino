import sys
from Batch import Batch

def main():
    args = sys.argv
    if len(args) == 3:
        models_folder = args[1]
        settings = args[2]
    else:
        models_folder = "example_models"
        settings = "example_settings.json"
    print(f'Running benchmarks for models from "{models_folder}" using settings from "{settings}"')
    batch = Batch(models_folder, settings)
    benchmarks = batch.benchmark_all_models()
    print(benchmarks)
    summary = batch.summarize()
    print("Benchmark summary created at", summary)

if __name__ == "__main__":
    main()
