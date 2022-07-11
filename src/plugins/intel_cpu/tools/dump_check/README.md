# CPU Dump Check Tool

Compile CPU plugin with `-DENABLE_DEBUG_CAPS=ON`, then this tool allows:

 - dump each output tensors from CPU plugin:
```bash
python3 cpu_dump_check.py -m=/path/to/model dump1
```

 - comparing two dumps and analyze differences:
```bash
python3 cpu_dump_check.py -m=/path/to/model dump1 dump2
```

 - visualize first error map:
```bash
python3 cpu_dump_check.py -m=/path/to/model dump1 dump2 -v
```


