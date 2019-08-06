# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name,too-many-locals,unused-variable
"""x86 nn operators"""
from __future__ import absolute_import as _abs
import tvm
from .. import generic

@generic.schedule_softmax.register(["cpu"])
def schedule_softmax(outs):
    '''Schedule for softmax

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of softmax
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    '''
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    softmax = outs[0]
    s = tvm.create_schedule([x.op for x in outs])

    op_tag = softmax.op.tag
    if op_tag == 'softmax_output':

        exp = softmax.op.input_tensors[0]
        expsum = softmax.op.input_tensors[1]
        max_elem = s[exp].op.input_tensors[1]
        axis = int(softmax.op.attrs['axis'])
        print("##########\nexp is {}\n expsum is {}\nmax_elem is {}\n axis is {}".format(exp, expsum, max_elem, axis))
        #fused_exp = s[exp].fuse(*s[exp].op.axis)
        #s[exp].parallel(s[exp].op.axis[0])
        s[exp].parallel(s[exp].op.axis[1])
        #s[exp].parallel(s[exp].op.axis[2])
        s[expsum].parallel(s[expsum].op.axis[0])
        s[max_elem].parallel(s[max_elem].op.axis[0])
        if len(s[max_elem].op.axis) > 1:
            s[max_elem].parallel(s[max_elem].op.axis[1])
        #s[max_elem].parallel(s[max_elem].op.axis[1])
        print("%%%%%%%%\n{}\n{}\nmax: {}\n".format(s[exp].op.axis, s[expsum].op.axis, s[max_elem].op.axis))
        """"
        exp is Tensor(shape=[1, 1024, 1024], op.name=tensor)
        expsum is Tensor(shape=[1, 1024], op.name=tensor)
        max_elem is Tensor(shape=[1, 1024], op.name=tensor)
        """
        axis is 1
    elif op_tag == 'log_softmax_output':
        exp = None
        max_elem = softmax.op.input_tensors[1]
        expsum = softmax.op.input_tensors[2]
        axis = 1
    else:
        raise ValueError('Tag is expected to be softmax_output or log_softmax_output. \
                         Got {0}'.format(op_tag))
    # only parallelize outer dimensions up to axis
    outer_axes = [s[softmax].op.axis[i] for i in range(0, axis)]
    fused_outer_axes = s[softmax].fuse(*outer_axes)
    s[softmax].parallel(fused_outer_axes)

    # move computations with the same outer dimensions under the same root
    s[max_elem].compute_at(s[softmax], fused_outer_axes)
    s[expsum].compute_at(s[softmax], fused_outer_axes)

    if exp != None:
        s[exp].compute_at(s[softmax], fused_outer_axes)

    return s

"""

def _fuse_parallel(s, x):
    if len(s[x].op.axis) >= 5:
        fused = s[x].fuse(s[x].op.axis[0], s[x].op.axis[1], s[x].op.axis[2])
        s[x].parallel(fused)
    elif len(s[x].op.axis) >= 3:
        fused = s[x].fuse(s[x].op.axis[0], s[x].op.axis[1])
        s[x].parallel(fused)
    else:
        s[x].parallel(s[x].op.axis[0])

@generic.schedule_softmax.register(["cpu"])
def schedule_softmax(outs):
    Schedule for softmax
    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of softmax
          in the format of an array of tensors.
    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
        [Tensor(shape=[1, 4], op.name=tensor), Tensor(shape=[1], op.name=tensor)]
        [Tensor(shape=[1, 1024, 1024], op.name=tensor), Tensor(shape=[1, 1024], op.name=tensor)]
                exp is Tensor(shape=[1, 1024, 1024], op.name=tensor)
        expsum is Tensor(shape=[1, 1024], op.name=tensor)
        max_elem is Tensor(shape=[1, 1024], op.name=tensor)
        axis is 1

    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    x = outs[0]
    print("$$$$$\n{}".format( x.op.input_tensors))
    max_elem = x.op.input_tensors[0]
    exp_sum = x.op.input_tensors[1]
    s = tvm.create_schedule([x.op for x in outs])
    tvm.schedule.AutoInlineInjective(s)
    for out in [max_elem, exp_sum, x]:
        _fuse_parallel(s, out)
    return s
"""