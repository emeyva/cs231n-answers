from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]

    for i in range(num_train):
        # calculate scores
        scores = X[i] @ W

        # avoid numerically instability - limited precision in floating-point arithmetic
        scores -= np.max(scores)

        # calculate probabilities
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores)

        # compute cross-entropy - only for the predicted probability corresponding to the correct class
        correct_class_prob = probs[y[i]]
        loss -= np.log(correct_class_prob)

        # Compute gradient for the softmax scores
        dW += np.outer(X[i], probs)


    # Average loss over all examples
    loss = loss / num_train + reg * np.sum(W**2)
    
    # Add gradient of regularization term
    dW = dW / num_train + 2 * reg * W  


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]

    # Compute scores
    scores = X @ W
    
    # avoid numerically instability - limited precision in floating-point arithmetic
    shifted_scores = scores - np.max(scores, axis=1, keepdims=True)
    
    # calculate probabilities
    probs = np.exp(shifted_scores)
    probs /= probs.sum(axis=1, keepdims=True)
    
    loss = -np.log(probs[range(num_train), y]).sum()
    
    # Add regularization to the loss
    loss = loss / num_train + reg * np.sum(W**2)
    
    # Compute gradient
    probs[range(num_train), y] -= 1  
    dW = X.T @ probs / num_train + 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
