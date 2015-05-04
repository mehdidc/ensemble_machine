import theano.tensor as T


def MDS_loss(dist_X, dist_Y):
    return T.sqrt(((dist_X - dist_Y) ** 2).sum(axis=1)).sum()


def euclidian_dist(X):
    s = (X.dimshuffle(0, 1, 'x') - X.dimshuffle('x', 1, 0)) ** 2
    return (s.sum(axis=1))


def hamming_dist(X):
    return T.neq(X.dimshuffle(0, 1, 'x'), X.dimshuffle('x', 1, 0)).sum(axis=1)
