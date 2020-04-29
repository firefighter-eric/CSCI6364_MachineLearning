def distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dist = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    print(dist)
    return dist

def mean(l):
    s_x = 0.0
    s_y = 0.0
    for x, y in l:
        s_x += x
        s_y += y
    m = (s_x/len(l), s_y/len(l))
    print(m)
    return m

A1 = (2, 10)
A2 = (2, 5)
A3 = (8, 4)
B1 = (5, 8)
B3 = (6, 4)
C1 = (1, 2)
C2 = (4, 9)

def test():
    a2a1 = distance(A2, A1)
    a2c1 = distance(A2, C1)

    c2a1 = distance(C2, A1)
    c2b1 = distance(C2, B1)

    m1 = mean([B1, C2, B3, A3])
    m2 = mean([C1, A2])

    b3m1 = distance(B3, m1)
    b3m2 = distance(B3, m2)

    c2m1 = distance(C2, m1)
    c2m2 = distance(C2, m2)

    m3 = mean([A1, C2])
    m4 = mean([B3, A3, B1])

    b1m3 = distance(B1, m3)
    b1m4 = distance(B1, m4)

l1 = [A1, A2, A3, B1, B3, C1, C2]
l2 = [A1, B1, C1]
for p in l1:
    for q in l2:
        distance(p, q)