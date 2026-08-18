# Intel GPU Regression Machine Inventory

Inventory captured before the M/C0/C1 builds on 2026-08-17 (Europe/Berlin).

## Source Baselines

- Fresh upstream master (M): `7a0a0b6b4655125b2d02ebc3c897b0ddfadbd310`
- Candidate branch head at initial C1 validation: `c136f152edaec9c43571ea4309375c8b307a873a`
- Latest candidate branch head: `7125f598962192bdc40dadf79155e40006d6f9eb`
- Original base and merge-base: `0f453eb8dca021e7176cdcc8570242c9f2fec7c5`
- Fresh-master integration candidate (C0): `3a2dfe0cd02d5bd8820140e39dc23eece517964e`
- Candidate after regression fixes (C1): `b094deb12936614a87f7f5dea413c5ebc9aaee57`
- Final latest-head integration (C2): `e59b4712f65b7dde43aecb12fe55bd1f3696bcd1`
- Final candidate with Vulkan recovery (C3): `a723a82cf828e6302209a13a380e2a1ec7e6dda8`

## Host and Toolchain

- GTA address and hostname: `10.190.233.128`, `DUT2198BMGFRD`
- OS/kernel: Ubuntu 24.04.3 LTS, x86_64, `7.0.0-29-generic`
- CPU: 13th Gen Intel Core i5-13400; 10 cores, 16 logical CPUs, 1 socket
- Build root: `/home/gta/intel_gpu_regression` with independent source and build directories
- Build tools: CMake 3.28.3, GCC/G++ 13.3.0, Ninja 1.11.1
- Proxy: `/home/gta/proxy.env`; Intel DMZ HTTP `:911` and HTTPS `:912`. APT and Git use matching persistent proxy settings. `curl` and `git ls-remote` checks against GitHub passed.

## GPU and Driver Inventory

| Role | PCI / render node | PCI ID | Runtime identity | Kernel driver |
|---|---|---|---|---|
| iGPU | `0000:00:02.0` / `renderD128` | `8086:4682` | Intel UHD Graphics 730 | `i915` |
| dGPU | `0000:03:00.0` / `renderD129` | `8086:e20b` | Intel Arc B580 Graphics | `xe` |

- Intel Compute Runtime: `26.27.39122.12-1~24.04~ppa1`; IGC: `2.38.3-3~24.04`.
- Mesa Vulkan: `25.2.8-0ubuntu0.24.04.2`; Vulkan loader/headers: 1.3.275.
- OpenCL enumerates the dGPU first and the iGPU second, as separate Intel platforms. Vulkan enumerates Arc B580 as physical device 0 and UHD 730 as physical device 1. OpenVINO public device IDs are not inferred from those orders and will be recorded per build.
- The fresh-master OCL build exposes `GPU.0` as the UHD 730 iGPU (`0000:00:02.0`) and `GPU.1` as the Arc B580 dGPU (`0000:03:00.0`). `AVAILABLE_DEVICES` is `0 1`; this is the legacy API contract used for M/C0/C1 comparison.
- Vulkan device UUIDs: dGPU `86800be2-0000-0000-0300-000000000000`; iGPU `86808246-0c00-0000-0002-000000000000`.
- `VK_LAYER_KHRONOS_validation` 1.3.275 is installed and discoverable.

Both devices have working OpenCL and Vulkan enumeration after the kernel update. The `xe` driver logs repeated `PCODE Mailbox failed: -6 Illegal Command` messages and cannot read the B580 power limits; GuC 70.44.1 is loaded while 70.54.0 is recommended. This is retained as a hardware-stack caveat for performance interpretation; no reset or firmware workaround was applied.
