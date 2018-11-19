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
  num_train = X.shape[0]
  num_classes = W.shape[1]
  for i in range(num_train):
    correct_class = y[i]

    # Softmax calculation
    scores = np.dot(X[i],W)
    scores -= np.max(scores) # For numeric stability. Visit: http://cs231n.github.io/linear-classify/#softmax
    exp_scores = np.exp(scores)
    softmax = exp_scores / np.sum(exp_scores)

    # Cross-entropy loss and derivative calculation
    for class_ in range(num_classes):
      if class_ == correct_class:
        loss += -np.log(softmax[class_])
        dW[:, class_] += X[i] * (softmax[class_] - 1)
      else:
        dW[:, class_] += X[i] * softmax[class_]

  loss /= num_train
  dW /= num_train

  loss += reg * np.sum(W * W)
  dW += reg * 2 * W
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

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
  num_train = X.shape[0]

  # Softmax calculation
  scores = X.dot(W)
  scores -= np.max(scores, axis=1, keepdims=True)
  exp_scores = np.exp(scores)
  softmax = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

  # Cross-entropy calculation
  indices = np.arange(num_train)
  loss = -np.log(softmax[indices, y])
  loss = np.sum(loss)

  # dW calculation
  softmax_editted = softmax
  softmax_editted[indices, y] -= 1
  dW = np.dot(X.T, softmax_editted)

  loss /= num_train
  dW /= num_train

  loss += reg * np.sum(W * W)
  dW += reg * 2 * W
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

