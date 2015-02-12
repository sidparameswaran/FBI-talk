from __future__ import division
import numpy as np
from SimplePEPS.tools import hexplot

data = {}
for i in xrange(-2, 2):
    for j in xrange(-1, 3):
        for k in xrange(0, 2):
            if -2<(i+j+k)<3 and -2<j<3 and -3<i<2:
                data[(i, j, k)] = 100*np.random.rand()
                
hexplot.hexplot(data)