from math import ceil, floor

class GRID(object):
    def __init__(self):
        self.threads_x     = 32 
        self.threads_y     = 12
        self.min_blocks    = 5

    def __str__(self):
        return 'Grid object has ({}, {}) blocks and ({}, {}) threads per block'.format(self.blocks_x, self.blocks_y, self.threads_x, self.threads_y)

    def blockAlloc(self, n, multiplier):
        tbp         = self.threads_x
        b           = self.min_blocks

        self.blocks_x = int(min(35, max(b, floor((2.0*n)/tbp))))
        self.blocks_y = min(35, 5*self.blocks_x)
        # self.blocks_y = int(min(30, max(b, floor((n*multiplier)/tbp)))) - self.blocks_x

        return self.blocks_x, self.blocks_y

