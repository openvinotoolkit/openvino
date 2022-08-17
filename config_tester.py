# type: ignore
from subprocess import run
from configs import getAllConfigs, benchmark_app_cpp, benchmark_app_py
from typing import Dict, List
import os

all_benches:Dict[str,List[str]] = {"cpp": [], "py": []}

def cmp(cppc, pyc):

    i = 0

    for bench_pair in zip(all_benches["cpp"], all_benches["py"]):
        print("===============================")
        for pair in zip(bench_pair[0], bench_pair[1]):
            pair_one = pair[0].replace("\n","")
            pair_two = pair[1].replace("\n","")

            if pair_one == pair_two:
                print(pair_one)
            else:
                print(f"{pair_one} ||||| {pair_two}")
                print(f"Config: {i}\n{cppc[i]}\n\n{pyc[i]}")
                exit(0)
        
        i+=1


def benchmarkTheBenchmark(configs:List[str], amount:int, name:str = "") -> None:
    
    for idx,command_line in enumerate(configs):
        if idx==amount: break

        with open(f"stuff{name}.txt","w") as file:
            run(command_line, shell=True, check=True, stdout=file, env = os.environ.copy())
        
        with open(f"stuff{name}.txt","r") as file:
            stuff = file.readlines()
            print("".join(stuff))
            all_benches[name].append(stuff)

(cpp_confs, i), (py_confs, _) = getAllConfigs(benchmark_app_cpp), getAllConfigs(benchmark_app_py)
benchmarkTheBenchmark(cpp_confs, i, "cpp")
benchmarkTheBenchmark(py_confs,  i, "py")
cmp(cpp_confs,py_confs)
