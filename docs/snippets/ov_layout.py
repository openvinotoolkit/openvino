# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# ! [ov:layout:simple]
from openvino.runtime import Layout
layout = Layout('NCHW')
# ! [ov:layout:simple]
# ! [ov:layout:complex]
# Each dimension has name separated by comma
# Layout is wrapped with square brackets
layout = Layout('[time,temperature,humidity]')
# ! [ov:layout:complex]
# ! [ov:layout:partially_defined]
# First dimension is batch, 4th is 'channels'.
# Others are not important for us
layout = Layout('N??C')

# Or the same using advanced syntax
layout = Layout('[n,?,?,c]')
# ! [ov:layout:partially_defined]
# ! [ov:layout:dynamic]
# First dimension is 'batch' others are whatever
layout = Layout('N...')

# Second dimension is 'channels' others are whatever
layout = Layout('?C...')

# Last dimension is 'channels' others are whatever
layout = Layout('...C')
# ! [ov:layout:dynamic]

# ! [ov:layout:predefined]
from openvino.runtime import layout_helpers
# returns 0 for batch
layout_helpers.batch_idx(Layout('NCDHW'))

# returns 1 for channels
layout_helpers.channels_idx(Layout('NCDHW'))

# returns 2 for depth
layout_helpers.depth_idx(Layout('NCDHW'))

# returns -2 for height
layout_helpers.height_idx(Layout('...HW'))

# returns -1 for width
layout_helpers.width_idx(Layout('...HW'))
# ! [ov:layout:predefined]

# ! [ov:layout:dump]
layout = Layout('NCHW')
print(layout)    # prints [N,C,H,W]
# ! [ov:layout:dump]
