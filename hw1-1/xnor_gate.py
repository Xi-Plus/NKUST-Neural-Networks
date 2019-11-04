# coding: utf-8
from nor_gate import NOR
from and_gate import AND
from or_gate import OR


def XOR(x1, x2):
    s1 = NOR(x1, x2)
    s2 = AND(x1, x2)
    y = OR(s1, s2)
    return y


if __name__ == '__main__':
    for xs in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        y = XOR(xs[0], xs[1])
        print(str(xs) + " -> " + str(y))
