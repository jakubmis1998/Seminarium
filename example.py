"""public static void intMaskMultiThread(int[][] m, int[][] result, int[][] mask, int X, int Y, int R) {
    int[][] pola = new int[X][Y];
    Ring halfRing = new Ring(R);
    IntStream.range(0, X).parallel().forEach(x0 -> {
        int lxbound = Math.max(0, x0 - R);
        int rxbound = Math.min(X, x0 + R + 1);
        IntStream.range(0, Y).forEach(y0 -> {
            IntStream.range(lxbound, rxbound).forEach(x -> {
                final int dx = x0 - x;
                final int dRx = dx + R;
                final int ry = halfRing.getHalfRing(dx);
                IntStream.range(Math.max(0, y0 - ry + 1), Math.min(Y, y0 + ry)).forEach(y -> {
                    pola[x0][y0]++;
                    final int dry = y + R - y0;
                    result[x0][y0] += ((m[x0][y0] - m[x][y]) >> 31) * mask[dRx][dry];
                });
            });
        });
    });
}"""

import numpy as np

def intMaskMultiThread(m, result, mask, X, Y, R):
    """
    m - macierz wejsciowa X x Y
    result - wyjscie
    mask - macierz wag 2R+1 x 2R+1
    R - promien otoczenia
    """
    pola = np.empty([X, Y], dtype=int)
    half_ring = R # cos zle

    for x0 in range(X):
        lxbound = max(0, x0 - R)
        rxbound = max(X, x0 + R + 1)
        for y0 in range(Y):
            for x in range(lxbound, rxbound):
                dx = x0 - x
                dRx = dx + R
                ry = dx # cos zle
                for y in range(max(0, y0 - ry + 1), min(Y, y0 + ry)):
                    dry = y + R - y0
                    pola[x0][y0] += mask[dRx][dry]
                    result[x0][y0] += ((m[x0][y0] - m[x][y]) >> 31) * mask[dRx][dry]

if __name__ == "__main__":
    for i in range(10):
        print(i)