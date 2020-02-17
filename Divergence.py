
import argparse
import tensorflow as tf
import types
import random
import numpy
from numpy import linalg as LA
import math
numpy.set_printoptions(precision=3, suppress=True)


def alloc(mode, hessians, alphas, hparams):
    graph = tf.Graph()
    session = tf.Session(graph=graph)
    n = hparams.n_nodes

    # Build optimization problem.
    with graph.as_default():

        # Hessians: H: Hessian contribution for each node.
        # H = [n x (n x n) ]
        H = []
        for h_i in hessians:
            H.append(tf.constant(h_i, tf.float32))

        # Hessians: H: Hessian contribution for each node.
        # A = [n]
        A = tf.constant(alphas, tf.float32)

        # Weights: W: Weight matix bids.
        # W = [n x n]
        w_list = []
        for _ in range(n):
            w_i = tf.Variable(tf.random.uniform((1, n), minval=0))
            w_list.append(w_i)
        W_concat = tf.concat(w_list, axis=0) # Build full matrix.
        W_sig = tf.sigmoid(W_concat) # Force onto [0,1]
        W = tf.linalg.normalize(W_sig, ord=1, axis=1)[0] # Sum to 1.

        # Weights_diag: W_dg: Matrix of W's main diagonal. a.k.a self-contribution.
        # W_dg = [n x n]
        W_dg = tf.matrix_set_diag(tf.zeros((n,n)), tf.linalg.tensor_diag_part(W), k = 0)

        # Interranking: Q: The inter-model ranking derived from the weights.
        # Q = [n x n]
        # We use the infinite series absorbing markov chain calculation.
        Q = tf.linalg.inv(tf.eye(hparams.n_nodes) - W + W_dg)
        Q = tf.linalg.normalize(Q, ord=1, axis=1)[0]

        # Mask: M: The mask to apply over inputs F.
        # Q = [n x n]
        shift = tf.reduce_mean(W, axis=0)
        M = tf.clip_by_value(tf.nn.relu(W - shift)*10, 0, 1)

        # Loss: L: The change to each loss effected by the Mask.
        # L = [n]
        l_list = []
        for i in range(n):
            h_i = H[i] # n x n
            m_i = tf.transpose(tf.slice(M, [i, 0], [1, -1])) # n x 1
            temp = tf.matmul(h_i, m_i) # n x 1
            l_i = tf.reshape(tf.reduce_sum(0.5 * temp * m_i), [1])
            l_list.append(l_i)
        L = tf.concat(l_list, axis=0)

        # Divergence score: D: Divergence of each ranking from mean.
        # D = [n]
        Q_avg = tf.reshape(tf.tile(tf.reduce_mean(Q, axis=0), [n]), [n,n])
        cross_entropy = -tf.reduce_sum(tf.multiply(Q_avg, tf.log(Q)), axis=1)
        D = tf.nn.softmax(tf.reshape(cross_entropy, [n]))

        # Utility: U: The utility gained or lost via the loss.
        U = tf.multiply(A, L)

        # Ranking: R : The ranking score.
        #R = tf.nn.softmax(tf.linalg.tensor_diag_part(Q * W_dg)) , ord=1, axis=0)[0]
        #R = tf.nn.softmax(tf.reduce_sum(W, axis=0))
        QQ = tf.reduce_sum(Q, axis=0, keepdims=True)
        RR = tf.div(tf.subtract(QQ, tf.reduce_min(QQ)), tf.subtract(tf.reduce_max(QQ), tf.reduce_min(QQ)))
        R = tf.squeeze(tf.linalg.normalize(RR, ord=1, axis=1)[0])

        # Payoff: P: U + R - D
        #P =  U #* alphas + R * (1 - alphas) * 0.001 # - D
        P = U + R

        ### Bellow Optimization.

        # Bidders move in the direction of the gradient of the Payoff.
        optimizer = tf.train.AdamOptimizer(hparams.learning_rate)

        # Mode == Competitive: All nodes optimize only their local payoff.
        train_steps = []
        if mode == 'competitive':
            for i in range(hparams.n_nodes):
                p_i = tf.slice(P, [i], [1])
                w_i = w_list[i]
                grads_and_vars_i = optimizer.compute_gradients(loss=-p_i, var_list=[w_i])
                train_steps.append(optimizer.apply_gradients(grads_and_vars_i))

        # Mode == Coordinated: Coordinated nodes optimize the aggregated payoff
        elif mode == 'coordinated':
            PP = U
            grads_and_vars = optimizer.compute_gradients(loss=-tf.reduce_mean(PP), var_list=w_list)
            train_steps.append(optimizer.apply_gradients(grads_and_vars))

        # Init the graph.
        session.run(tf.global_variables_initializer())

        # Converge...
        for step in range(hparams.n_steps):

            # Randomly choose participant to optimize
            if mode == 'competitive':
                step = random.choice(train_steps)

            # Optimize all participants.
            elif mode == 'coordinated':
                step = train_steps[0]

            # Run graph.
            output = session.run(fetches =
                                      {
                                        'step': step,
                                        'P': P,
                                        'U': U,
                                        'R': R,
                                        'D': D,
                                        'W': W,
                                        'M': M,
                                        'Q': Q,
                                      })
        # Return metrics.
        return output

def kl(p, q):
    """Kullback-Leibler divergence D(P || Q) for discrete distributions
    Parameters
    ----------
    p, q : array-like, dtype=float, shape=n
    Discrete probability distributions.
    """
    p = numpy.asarray(p, dtype=numpy.float)
    q = numpy.asarray(q, dtype=numpy.float)

    return numpy.sum(numpy.where(p != 0, p * numpy.log(p / (q + 0.0001)), 0))

# Hessians: H: The hessian of the loss w.r.t a change in weights.
# via second order taylor series approximation. First term is 0 at convergence.
# Second term is parameterized by the Hessian term.
# ∆L = M^t * H * M
def make_hessians(n, size):
    assert(size > 1)
    assert(n > 0)
    hessians = []
    for i in range(n):
        h_i = numpy.random.randn(size, size)
        h_i = (h_i - numpy.min(h_i))/numpy.ptp(h_i)
        h_i = h_i/h_i.sum(axis=1, keepdims=1)
        hessians.append(h_i)
    return hessians

# Alphas: A: The trade off between optimizing Utility vs optimizing for
# ranking. P = α M * H * M + R
def make_alphas(n):
    alphas = numpy.random.randn(n)
    alphas = (alphas - numpy.min(alphas))/numpy.ptp(alphas)
    return alphas

def softmax(scores):
    zeta = 0
    for s in scores:
        zeta += math.exp(s)
    softmax = []
    for s in scores:
        softmax.append(math.exp(s)/zeta)
    return softmax

def idealized_ranking(hparams, hessians):
    scores = []
    for i in range(hparams.n_nodes):
        score_i = 0
        delta_i = -numpy.eye(hparams.n_nodes)[i,:]
        for h_j in hessians:
            score_i += numpy.dot(delta_i, numpy.matmul(h_j, delta_i))
        scores.append(score_i)
    scores = numpy.asarray(scores)
    scores = (scores - numpy.min(scores))/numpy.ptp(scores)
    print (scores)
    scores = scores/numpy.sum(scores)
    return scores


def dcg_at_k(r, k):
    r = numpy.asfarray(r)[:k]
    if r.size:
        return numpy.sum(numpy.subtract(numpy.power(2, r), 1) / numpy.log2(numpy.arange(2, r.size + 2)))
    return 0.


def ndcg_at_k(r, k):
    idcg = dcg_at_k(sorted(r, reverse=True), k)
    if not idcg:
        return 0.
    return dcg_at_k(r, k) / idcg


def trial(hparams):
    # Hessians: H: The hessian of the loss w.r.t a change in weights.
    hessians = make_hessians(hparams.n_nodes, hparams.n_nodes)

    # Alphas: Lipshitz constant of utility.
    alphas = make_alphas(hparams.n_nodes)

    # Run coordinated weight convergence.
    coord_output = alloc('coordinated', hessians, alphas, hparams)

    comp_output = alloc('competitive', hessians, alphas, hparams)

    print ('Coordinated Weights: W')
    print (coord_output['W'])
    print ('')

    print ('Competitive Weights: W')
    print (comp_output['W'])
    print ('')

    print ('Coordinated Mask: M')
    print ('M = σ ( W - avg(W) )')
    print (coord_output['M'])
    print ('')

    print ('Competitive Mask: M')
    print ('M = σ ( W - avg(W) )')
    print (comp_output['M'])
    print ('')

    print ('Coordinated Interranking: Q')
    print ('Q = (I - W + Wdg)')
    print (coord_output['Q'])
    print ('')

    print ('Competitve Interranking: Q')
    print ('Q = (I - W + Wdg)')
    print (comp_output['Q'])
    print ('')

    # Alphas: A: The trade off between optimizing Utility vs optimizing for
    # ranking. P = αU + (1- α)R
    print ('Alphas: A')
    print ('U = α MT H MT')
    print (alphas)
    print ('')

    print ('Coordinated Ranking: R')
    print ('softmax( Q * Wdg)')
    print (coord_output['R'])
    print ('')

    print ('Competitive Ranking: R')
    print ('softmax( Q * Wdg)')
    print (comp_output['R'])
    print ('')

    print ('Coordinated Utility: U')
    print ('U = A o (M^t * H * M)')
    print (coord_output['U'])
    print ('')

    print ('Competitive Utility: U')
    print ('U = A o (M^t * H * M)')
    print (comp_output['U'])
    print ('')

    print ('Coordinated Payoff: P')
    print ('P = U + R')
    print (coord_output['P'])
    print ('Avg:' + str(sum(coord_output['P'])/hparams.n_nodes))
    print ('')

    print ('Competitive Payoff: P')
    print ('P = U + R')
    print (comp_output['P'])
    print ('Avg:' + str(sum(comp_output['P'])/hparams.n_nodes))
    print ('')

    coord_sparsity = numpy.count_nonzero(coord_output['M'])/coord_output['M'].size
    print ('Coordinated Mask Sparsity: ' + str(coord_sparsity))
    print ('')

    comp_sparsity = numpy.count_nonzero(comp_output['M'])/comp_output['M'].size
    print ('Competitive Mask Sparsity: ' + str(comp_sparsity))
    print ('')

    ideal_rank = idealized_ranking(hparams, hessians)
    print (ideal_rank)
    print (comp_output['R'])
    print (coord_output['R'])

    print (numpy.argsort(ideal_rank))
    print (numpy.argsort(comp_output['R']))
    print (numpy.argsort(coord_output['R']))

    divergence1 = kl(ideal_rank, comp_output['R'])
    print ('Ranking KL Divergence: ' + str(divergence1))
    print ('')







def main(hparams):
    for _ in range(hparams.trials):
        trial(hparams)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.ERROR)
    graph = tf.Graph()
    session = tf.Session(graph=graph)
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--trials',
        default=1,
        type=int,
        help="Number of trials to run.")
    parser.add_argument(
        '--n_steps',
        default=1000,
        type=int,
        help="Number of convergence steps to run.")
    parser.add_argument(
        '--learning_rate',
        default=0.05,
        type=float,
        help="Optimizer Learning rate")
    parser.add_argument(
        '--n_nodes',
        default=10,
        type=int,
        help="Number of nodes to simulate.")

    hparams = parser.parse_args()

    main(hparams)
