import numpy as np

from ..layers import *
from ..layer_utils import *

# from assignment2.cs231n.layer_utils import *
# from assignment2.cs231n.layers import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3 * 32 * 32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        ############################################################################
        W1 = np.random.normal(0, weight_scale, (input_dim, hidden_dim))
        b1 = np.zeros((hidden_dim,))
        W2 = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
        b2 = np.zeros((num_classes,))
        self.params['W1'] = W1
        self.params['b1'] = b1
        self.params['W2'] = W2
        self.params['b2'] = b2

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        N, D = X.shape
        W1 = self.params['W1']
        b1 = self.params['b1']
        W2 = self.params['W2']
        b2 = self.params['b2']

        hidden_linear = np.dot(X, W1) + b1
        hidden_activation = np.maximum(hidden_linear, 0)    # (N, hidden_dim)
        scores = np.dot(hidden_activation, W2) + b2

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        scores -= np.max(scores, axis=1).reshape((N, 1))
        exp_scores = np.exp(scores)
        softmax_probs = exp_scores / np.sum(exp_scores, axis=1).reshape(N, 1)
        loss = np.sum(-np.log(softmax_probs[np.arange(N), y]))
        loss /= N
        loss += 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2))

        score_grads = np.copy(softmax_probs)
        score_grads[np.arange(N), y] -= 1   # (N, num_classes)
        dW2 = np.dot(hidden_activation.T, score_grads) / N  # (hidden_dim, num_classes)
        dW2 += self.reg * W2
        # dW2 += self.reg * 2 * W2
        db2 = np.sum(score_grads, axis=0) / N

        hidden_activation_grads = np.dot(score_grads, W2.T)
        hidden_linear_grads = np.copy(hidden_activation_grads)
        hidden_linear_grads[ hidden_linear < 0 ] = 0
        dW1 = np.dot(X.T, hidden_linear_grads) / N
        dW1 += self.reg * W1
        db1 = np.sum(hidden_linear_grads, axis=0) / N

        grads['W2'], grads['b2'] = dW2, db2
        grads['W1'], grads['b1'] = dW1, db1

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3 * 32 * 32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        self.caches = {}
        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        ############################################################################
        self.params['W1'] = np.random.normal(0, weight_scale, (input_dim, hidden_dims[0]))
        self.params['b1'] = np.zeros((hidden_dims[0], ))
        if self.use_batchnorm:
            self.params['gamma1'] = np.random.randn(hidden_dims[0])
            self.params['beta1'] = np.random.randn(hidden_dims[0])

        for l in xrange(1, len(hidden_dims)):
            self.params['W' + str(l+1)] = np.random.normal(0, weight_scale, (hidden_dims[l-1], hidden_dims[l]))
            self.params['b' + str(l+1)] = np.zeros((hidden_dims[l], ))
            if self.use_batchnorm:
                self.params['gamma' + str(l+1)] = np.random.randn(hidden_dims[l])
                self.params['beta' + str(l+1)] = np.random.randn(hidden_dims[l])

        self.params['W' + str(self.num_layers)] = np.random.normal(0, weight_scale, (hidden_dims[-1], num_classes))
        self.params['b' + str(self.num_layers)] = np.zeros((num_classes, ))

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.dropout_param is not None:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        linear = None
        x = X
        activation_outs = [x]
        for i in xrange(self.num_layers):
            cur_layer = i + 1                       # The layer pointer
            W = self.params['W' + str(cur_layer)]
            b = self.params['b' + str(cur_layer)]

            affine_out, affine_cache = affine_forward(x, W, b)
            self.caches['affine_cache' + str(cur_layer)] = affine_cache
            if cur_layer != self.num_layers:
                if self.use_batchnorm:
                    gamma = self.params['gamma' + str(cur_layer)]
                    beta = self.params['beta' + str(cur_layer)]
                    bn_param = self.bn_params[cur_layer-1]
                    affine_out, bn_cache = batchnorm_forward(affine_out, gamma, beta, bn_param)
                    self.caches['bn_cache' + str(cur_layer)] = bn_cache
                activation_out, activation_cache = relu_forward(affine_out)
                self.caches['activation_cache' + str(cur_layer)] = activation_cache
                if self.use_dropout:
                    activation_out, dropout_cache = dropout_forward(activation_out, self.dropout_param)
                    self.caches['dropout_cache' + str(cur_layer)] = dropout_cache
                activation_outs.append(activation_out)
                x = activation_out
            else:
                linear = affine_out
        scores = linear if linear is not None else None
        # print self.caches.keys()
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        # print self.caches.keys()
        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, scores_grad = softmax_loss(x=scores, y=y)
        reg_loss = 0.0
        for l in xrange(1, self.num_layers + 1):
            W = self.params['W' + str(l)]
            reg_loss += np.sum(W * W)
        reg_loss *= (0.5 * self.reg)
        loss += reg_loss

        # hidden layers
        grad = scores_grad
        for i in xrange(self.num_layers):
            cur_layer = self.num_layers - i
            x = activation_outs[cur_layer - 1]
            w = self.params['W' + str(cur_layer)]
            b = self.params['b' + str(cur_layer)]
            # dx, dw, db = affine_backward(grad, cache=(x, w, b))
            dx, dw, db = affine_backward(grad, cache=self.caches['affine_cache' + str(cur_layer)])
            dw += self.reg * w
            grads['W' + str(cur_layer)] = dw
            grads['b' + str(cur_layer)] = db
            # update the grad to last layer(self.num_layers - l - 1). The input layer does not have activation
            if cur_layer - 1 > 0:
                if self.use_dropout:
                    dx = dropout_backward(dx, self.caches['dropout_cache' + str(cur_layer - 1)])
                # grad = relu_backward(dx, cache=x)
                grad = relu_backward(dx, cache=self.caches['activation_cache' + str(cur_layer - 1)])

            if self.use_batchnorm and 'bn_cache' + str(cur_layer - 1) in self.caches:
                dx, dgamma, dbeta = batchnorm_backward(grad, self.caches['bn_cache' + str(cur_layer - 1)])
                grads['gamma' + str(cur_layer - 1)] = dgamma
                grads['beta' + str(cur_layer - 1)] = dbeta
                grad = dx

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

if __name__ == '__main__':
    fcnet = FullyConnectedNet([15, 20, 15], use_batchnorm=True)
    for name, param in fcnet.params.iteritems():
        print name, param.shape
    print fcnet.num_layers
    print fcnet.bn_params
    print fcnet.dropout_param