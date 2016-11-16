import numpy as np
from random import shuffle


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in xrange(num_train):
        scores = X[i].dot(W)
        # scores = W.dot(X[:, i])
        correct_class_score = scores[y[i]]
        # margins = np.maximum(0, scores - scores[y[i]] + 1) # note delta = 1
        # margins[correct_class_idx] = 0
        # loss += np.sum(margins)
        diff_count = 0
        for j in xrange(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                diff_count += 1
                # print X[i].shape, scores[j], correct_class_score, margin, loss, (X[i] * margin).shape, dW[j, :].shape
                # dW[:, j] += X[i] * margin  # hyj
                dW[:, j] += X[i]    # gradient update for incorrect rows
        # gradient update for correct row
        dW[:, y[i]] += -diff_count * X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    # Add regularization to the loss.
    loss += 0.5 * reg * np.sum(W * W)

    dW /= num_train  # hyj
    dW += reg * W  # hyj

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # W shape: (num_feature, num_class)
    # X shape: (num_train, num_feature)
    # scores shape: (num_train, num_class), so as correct_class_scores, margins
    num_train = X.shape[0]
    num_feature = X.shape[1]
    num_class = W.shape[1]

    scores = X.dot(W)
    correct_class_score = scores[np.arange(num_train), y]
    margins = np.maximum(0, (scores.T - correct_class_score).T + 1)
    # margins = np.maximum(0, scores - correct_class_score[:, np.newaxis] + 1)
    margins[np.arange(num_train), y] = 0

    loss += np.sum(margins) / num_train
    loss += 0.5 * reg * np.sum(W*W)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################


    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################













    # Semi-vectorized version. It's not any faster than the loop version.
    # num_classes = W.shape[1]
    # incorrect_counts = np.sum(margins > 0, axis=1)
    # for k in xrange(num_classes):
    #   # use the indices of the margin array as a mask.
    #   dwj = np.sum(X[margins[:, k] > 0], axis=0)
    #   dwy = np.sum(-incorrect_counts[y == k][:, np.newaxis] * X[y == k], axis=0)
    #   dW[:, k] = dwj + dwy

    # Fully vectorized version. Roughly 10x faster.
    X_mask = np.zeros(margins.shape)
    # column maps to class, row maps to sample; a value v in X_mask[i, j]
    # adds a row sample i to column class j with multiple of v
    X_mask[margins > 0] = 1
    # for each sample, find the total number of classes where margin > 0
    incorrect_counts = np.sum(X_mask, axis=1)
    # print X_mask.shape, incorrect_counts.shape
    X_mask[np.arange(num_train), y] = -incorrect_counts
    dW = X.T.dot(X_mask)
    dW /= num_train
    dW += reg * W

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return loss, dW
