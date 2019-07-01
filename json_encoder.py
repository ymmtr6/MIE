# -*- coding:utf-8 -*-

import numpy
import json


class MyEncoder(json.JSONEncoder):

    def default(self, o):  # pylint: disable=E0202
        if isinstance(o, numpy.array):
            return o.tolist()
        elif isinstance(o, numpy.ndarray):
            return o.tolist()
        else:
            return super(MyEncoder, self).default(o)
