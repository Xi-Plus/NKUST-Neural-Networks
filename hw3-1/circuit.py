class Circuit():
    def __init__(self, a, x, b, y, c):
        self.A = a
        self.X = x
        self.B = b
        self.Y = y
        self.C = c

    def backward(self, loss):
        self.c = loss * 1
        self.axpby = loss * 1
        self.ax = self.axpby * 1
        self.by = self.axpby * 1
        self.a = self.ax * self.X
        self.x = self.ax * self.A
        self.b = self.by * self.Y
        self.y = self.by * self.B
