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
    correct_class_scores = scores[xrange(num_train), y]
    margins = (scores.T - correct_class_scores).T + 1
    margins[xrange(num_train), y] = 0
    margin_valid_indices = margins > 0
    margins[ margins < 0] = 0

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

    # dW -> dW_tensor: (num_feature, num_class) -> (num_train, num_feature, num_class)
    dW_tensor = np.tile(dW, (num_train, 1, 1))
    # dW_tensor = np.tile(np.expand_dims(dW, axis=0), (num_train, 1, 1))

    # scores -> scores_tensor: (num_train, num_class) -> (num_train, num_class, num_feature)
    scores_tensor = np.tile(np.expand_dims(scores, axis=2), (1, 1, num_feature))

    # X -> X_tensor: (num_train, num_feature) -> (num_train, num_feature, num_class)
    X_tensor = np.tile(np.expand_dims(X, axis=2), (1, 1, num_class))

    # margins -> margin_tensor: (num_train, num_class) -> (num_train, num_feature, num_class)
    margin_tensor = np.tile(np.expand_dims(margins, axis=1), (1, num_feature, 1))

    dW_tensor = X_tensor[:] * margin_tensor[:]
    dW = np.sum(dW_tensor, axis=0)
    dW /= num_train
    dW += reg * W

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return loss, dW
