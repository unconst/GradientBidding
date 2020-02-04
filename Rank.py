
import argparse
import tensorflow as tf
import types
import random
import numpy
from numpy import linalg as LA
numpy.set_printoptions(precision=4, suppress=True)


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

        # Weights: W: Weight matix bids.
        # W = [n x n]
        w_list = []
        for _ in range(n):
            w_i = tf.Variable(tf.random.uniform((1, n), minval=0))
            w_list.append(w_i)
        W_concat = tf.concat(w_list, axis=0) # Build full matrix.
        W_sig = tf.sigmoid(W_concat) # Force onto [0,1]
        W = tf.linalg.normalize(W_sig, ord=1, axis=1)[0] # Sum to 1.

        # Mask: M: The mask to apply over inputs F.
        # Q = [n x n]
        shift = tf.reduce_mean(W, axis=0)
        M = tf.nn.relu6(W - shift)

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

        # Utility: U: The utility gained or lost via the loss.
        U = L

        # Ranking: R : The ranking score.
        #W_dg = tf.matrix_set_diag(tf.zeros((n,n)), tf.linalg.tensor_diag_part(W), k = 0)
        #W = W - W_dg
        #R = tf.squeeze(tf.linalg.normalize(tf.reduce_sum(W, axis=0, keepdims=True), ord=1, axis=1)[0])
        R = tf.nn.softmax(tf.reduce_sum(W, axis=0))


        # Divergence score: D: Divergence of each ranking from mean.
        # D = [n]
        W_avg = tf.reshape(tf.tile(tf.reduce_mean(W, axis=0), [n]), [n,n])
        cross_entropy = -tf.reduce_sum(tf.multiply(W_avg, tf.log(W)), axis=1)
        D = tf.nn.softmax(tf.reshape(cross_entropy, [n]))

        # Payoff: P: U + R - D
        #P =  U #* alphas + R * (1 - alphas) * 0.001 # - D
        #P = U + R * 5
        P = U # * alphas  - D * (1 - alphas)

        topk = tf.math.top_k(R, k=n, sorted=True)[1]

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
            grads_and_vars = optimizer.compute_gradients(loss=-tf.reduce_mean(P), var_list=w_list)
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
                                        'D': D,
                                        'R': R,
                                        'W': W,
                                        'M': M,
                                        'topk': topk,
                                      })

            print (output['M'])
            print (output['topk'])

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

    return numpy.sum(numpy.where(p != 0, p * numpy.log(p / q), 0))

def Rank(hparams, hessians):
    rank_j = []
    identity = numpy.eye(hparams.n_nodes)
    for j in range(hparams.n):
        mask = identity[i:-1]
        j_sum = 0
        for i in range(hparams.n_nodes):
            i_sum = 0
            for hess in Hessians:
                 cost_sum += numpy.sum(numpy.matmul(hess, mask))
            j_sum += i_sum
        rank_j.append(j_sum)
    return rank_j

def trial(hparams):
    # Hessians: H: The hessian of the loss w.r.t a change in weights.
    # via second order taylor series approximation. First term is 0 at convergence.
    # Second term is parameterized by the Hessian term.
    # ∆L = M^t * H * M
    print ('Hessians: H')
    print ('∆L = M^t * H * M')
    hessians = []
    for i in range(hparams.n_nodes):
        h_i = numpy.random.randn(hparams.n_nodes, hparams.n_nodes)
        h_i = (h_i - numpy.min(h_i))/numpy.ptp(h_i)
        h_i = h_i/h_i.sum(axis=1, keepdims=1)
        print (h_i)
        print ('')
        hessians.append(h_i)
    print ('')

    # Alphas: A: The trade off between optimizing Utility vs optimizing for
    # ranking. P = αU + (1- α)R
    print ('Alphas: A')
    print ('P = αU + (1- α)R')
    alphas = numpy.random.randn(hparams.n_nodes)
    alphas = (alphas - numpy.min(alphas))/numpy.ptp(alphas)
    print (alphas)
    print ('')

    # Run coordinated weight convergence.
    coord_output = alloc('coordinated', hessians, alphas, hparams)

    # Run competitive weight convergence.
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

    # Alphas: A: The trade off between optimizing Utility vs optimizing for
    # ranking. P = αU + (1- α)R
    print ('Alphas: A')
    print ('P = αU + (1- α)R')
    print (alphas)
    print ('')

    print ('Coordinated Divergence: D')
    print ('D = KL ( Q, avg(Q) )')
    print (coord_output['D'])
    print ('')

    print ('Competitive Divergence: D')
    print ('D = KL ( Q, avg(Q) )')
    print (comp_output['D'])
    print ('')

    print ('Coordinated Ranking: R')
    print ('softmax(sum(W, axis=0))')
    print (coord_output['R'])
    print ('')

    print ('Competitive Ranking: R')
    print ('softmax(sum(W, axis=0))')
    print (comp_output['R'])
    print ('')

    print ('Coordinated Utility: U')
    print ('U = M^t * H * M')
    print (coord_output['U'])
    print ('')

    print ('Competitive Utility: U')
    print ('U = M^t * H * M')
    print (comp_output['U'])
    print ('')

    print ('Coordinated Payoff: P')
    print ('P = A * U + (1 - A) * R')
    print (coord_output['P'])
    print ('Avg:' + str(sum(coord_output['P'])/hparams.n_nodes))
    print ('')

    print ('Competitive Payoff: P')
    print ('P = A * U + (1 - A) * R')
    print (comp_output['P'])
    print ('Avg:' + str(sum(comp_output['P'])/hparams.n_nodes))
    print ('')

    print ('Coordinated sorted: ' + str(coord_output['topk']))
    print ('')

    print ('Competitve sorted: ' + str(comp_output['topk']))
    print ('')

    divergence = kl(coord_output['R'], comp_output['R'])
    print ('Ranking KL Divergence: ' + str(divergence))
    print ('')

    coord_sparsity = numpy.count_nonzero(coord_output['M'])/coord_output['M'].size
    print ('Coordinated Mask Sparsity: ' + str(coord_sparsity))
    print ('')

    comp_sparsity = numpy.count_nonzero(comp_output['M'])/comp_output['M'].size
    print ('Competitive Mask Sparsity: ' + str(comp_sparsity))
    print ('')

    poa = sum(comp_output['P']) / sum(coord_output['P'])
    print ('Price of Anarchy: ' + str(poa))

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
