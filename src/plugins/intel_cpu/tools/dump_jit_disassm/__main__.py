import os
import re
import argparse
import platform
import subprocess
import sys
from colorama import Fore

def addr2line(exefile, addrs, addr2line_path):
    proc = subprocess.Popen([addr2line_path, "-C", "-e", exefile, "-f", "-s", "-p"],
                            stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8")
    line_info = {}
    for i, addr in enumerate(addrs):
        proc.stdin.write(f'{addr}\n')
        proc.stdin.flush()
        line_info[addr] = proc.stdout.readline()[:-1] # remove line end

    proc.stdin.close()
    proc.terminate()
    proc.wait(timeout=0.2)
    return line_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"jit dump disassmbler")
    parser.add_argument("--filter", type=list, default=["xbyak.h","jit_generator.hpp","primitive.hpp","xbyak_mnemonic.h"])
    parser.add_argument("--maxcnt", type=int, help="Maximum number of entries from call stack to show", default=3)
    parser.add_argument("--addr2line", type=str, help="path to addr2line tool", default="addr2line")
    parser.add_argument("trace", type=str)
    parser.add_argument("bin", type=str)

    args = parser.parse_args()

    def skip_src_file(file_lino):
        for filter in args.filter:
            if (file_lino.startswith(filter)):
                return True
        return False

    print(f"load {args.trace}...")
    addr2check = {}
    offset2addr = {}
    pattern = re.compile("(.*)\(\+(0x[0-9a-f]*)\).*")  # xxx.so(+0x1234) [] 
    pattern2 = re.compile("(.*)\(\) \[(0x[0-9a-f]*)\].*") # xxx() [0x1234]
    with open(args.trace, "r") as f:
        for line in f.readlines():
            offset, traces = line.split(":")
            offset = offset.strip()
            for trace in traces.split(",")[1:]:
                m = pattern.match(trace)
                if not m:
                    m = pattern2.match(trace)
                if (m):
                    exefile=m.group(1)
                    addr=m.group(2)
                    if not exefile in addr2check:
                        addr2check[exefile] = set()
                    if not offset in offset2addr:
                        offset2addr[offset] = []
                    addr2check[exefile].add(addr)
                    offset2addr[offset].append([exefile, addr])

    print(f"extract debug info with addr2line...")
    traces={}
    for exefile, addrs in addr2check.items():
        traces[exefile] = addr2line(exefile, addrs, args.addr2line)

    print("objdump...")
    if platform.machine() == "x86_64":
        disassemble = subprocess.check_output(f"objdump -D -b binary -mi386:x86-64 -M intel {args.bin}", shell=True, encoding="utf-8")
    elif platform.machine() == "aarch64":
        disassemble = subprocess.check_output(f"objdump -D -b binary -m aarch64 {args.bin}", shell=True, encoding="utf-8")
    else:
        print(f"Unsupported platform for objdump: {platform.machine()}")
        sys.exit(1)

    pattern = re.compile("^\s*([\da-f]*):.*")
    print("parsing...")

    rm_prefixes = [
        "dnnl::impl::cpu::x64::",
        "dnnl::impl::cpu::aarch64::",
    ]
    for line in disassemble.splitlines():
        m = pattern.match(line)
        debug_info = []
        if (m):
            offset = m.group(1)
            if offset in offset2addr:
                for exefile, addr in offset2addr[offset]:
                    t = traces[exefile][addr]
                    for rm_prefix in rm_prefixes:
                        if t.startswith(rm_prefix):
                            t = t[len(rm_prefix):]
                    file_lino = t.split(" at ")[1]
                    if not skip_src_file(file_lino):
                        debug_info.append(file_lino)
        if len(debug_info):
            print(line + "\t" + Fore.YELLOW + " / ".join(debug_info[:args.maxcnt]) + Fore.RESET)
        else:
            print(line)

