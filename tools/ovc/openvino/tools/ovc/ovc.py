#!/usr/bin/env python3

import sys

if __name__ == "__main__":
    from openvino.tools.ovc.telemetry_utils import init_mo_telemetry
    from openvino.tools.ovc.main import main

    init_mo_telemetry()
    sys.exit(main())
