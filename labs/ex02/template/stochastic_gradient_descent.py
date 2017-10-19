# -*- coding: utf-8 -*-
"""Stochastic Gradient Descent"""

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    e = y-tx.dot(w)
    return -1/(np.shape(y)[0])*tx.T.dot(e)


def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        loss = compute_loss(y, tx, w)
        (y_st, tx_st) = [batch for batch in batch_iter(y, tx, batch_size)][0]
        grad = compute_gradient(y_st, tx_st, w)
        w = w-gamma*grad
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter+1, ti=max_iters, l=loss, w0=w[0], w1=w[1]))
    return losses, ws