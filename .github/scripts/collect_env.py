#!/usr/bin/env python3
import platform
import sys
import os

def main():
    info = {
        "OS": platform.system(),
        "OS Release": platform.release(),
        "Architecture": platform.machine(),
        "Python Version": sys.version.split()[0],
        "Processor": platform.processor() or "Unknown",
        "CPU Count": str(os.cpu_count()),
    }
    
    md_lines = [
        "### 🛠️ Contributor Build/Test Environment Details",
        "",
        "Automated capture of the environment executing this Pull Request.",
        "",
        "| Component | Details |",
        "| --- | --- |"
    ]
    
    for k, v in info.items():
        md_lines.append(f"| **{k}** | `{v}` |")
        
    md_lines.extend([
        "",
        "> _This comment is generated automatically. It updates when new commits are pushed._"
    ])
    
    with open("env_info.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

if __name__ == "__main__":
    main()

