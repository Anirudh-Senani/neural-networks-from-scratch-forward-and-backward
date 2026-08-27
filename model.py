"""
Neural Networks From Scratch: Forward and Backward

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - numerical_gradient
def numerical_gradient(f, x, eps=1e-5):
    # TODO: Estimate the gradient of scalar f w.r.t. array x via central finite differences
    grad = np.zeros_like(x, dtype=np.float64)

    for ind in np.ndindex(*x.shape):
        x_plus = x.copy()
        x_plus[ind] += eps
        f_plus = f(x_plus)

        x_minus = x.copy()
        x_minus[ind] -= eps
        f_minus = f(x_minus)

        grad[ind] = (f_plus - f_minus)/(2.0*eps)

    return grad

# Step 2 - gradient_check
def gradient_check(analytic_grad, numeric_grad, tol=1e-5):
    # TODO: Return max relative error between analytic and numeric gradients.
    denom = np.maximum(np.abs(analytic_grad), np.abs(numeric_grad))
    denom = np.maximum(denom, tol)
    return np.max(np.abs(analytic_grad - numeric_grad)/denom)

# Step 3 - make_dense
def make_dense(in_dim, out_dim, weight_init_fn):
    """Create a fully connected layer.

    Inputs:
      in_dim: int, input feature size
      out_dim: int, output feature size
      weight_init_fn: callable(in_dim, out_dim) -> (W, b)

    Returns layer dict with keys:
      params: {'W': (in_dim, out_dim), 'b': (out_dim,)}
      forward(x) -> (y, cache) with y shape (batch, out_dim)
      backward(dout, cache) -> (dx, grads) with grads {'W', 'b'}
        Analytic dx/dW/db must match numerical_gradient via gradient_check.
    """
    # TODO: your approach here
    W, b = weight_init_fn(in_dim, out_dim)


    def forward(x):
        cache = {'W': W, 'x': x}
        out = x @ W + b
        return out, cache


    def backward(dout, cache):
        dW = cache['x'].T @ dout
        dx = dout @ cache['W'].T
        db = dout.sum(axis=0)
        return dx, {'W':dW, 'b':db}


    return dict(
        params={'W': W, 'b' : b},
        forward=forward,
        backward=backward
    )

# Step 4 - make_activation
def make_activation(kind='relu'):
    """Create a genuinely nonlinear elementwise activation layer.

    Args:
        kind: str nonlinearity name. Default 'relu' must implement ReLU
              (zero negatives, pass non-negatives). Other kinds optional.

    Returns:
        Layer dict with:
          forward(x) -> (y, cache)
            x, y: np.ndarray shape (batch, dim)
          backward(dout, cache) -> (dx, {})
            dout, dx: np.ndarray shape (batch, dim)
            param grad dict is always empty (no learnable params)

    Must be elementwise and non-affine; analytic dx must match
    numerical_gradient / gradient_check.
    """
    # TODO: your approach here
    if kind == 'relu':
        def forward(x):
            return np.maximum(x, 0.0), {'x':x}


        def backward(dout, cache):
            dx = np.where(cache['x']>0.0, dout, 0.0)
            return dx, {}
    elif kind == 'tanh':
        def forward(x):
            a = np.tanh(x)
            return a, {'a': a}


        def backward(dout, cache):
            dx = 1 - cache['a']**2
            return dx, {}
    else:
        def forward(x):
            return x, {}


        def backward(dout, cache):
            return dout, {}


    return dict(
        params={},
        forward=forward,
        backward=backward
    )

# Step 5 - initialize_weights
def initialize_weights(in_dim, out_dim, scheme='he'):
    """Return (W, b) for a dense layer.

    Inputs:
      in_dim: int fan-in
      out_dim: int fan-out
      scheme: str initialization family (default 'he')

    Returns:
      W: np.ndarray shape (in_dim, out_dim), finite, symmetry-breaking,
         scale stable with depth (fan-in dependent)
      b: np.ndarray shape (out_dim,), near zero
    """
    # TODO: your approach here
    if scheme == 'he':
        scale = (2/in_dim)**0.5
        W = np.random.normal(loc=0.0, scale=scale, size=(in_dim, out_dim))
        b = np.random.randn(out_dim) * 0.0001
    elif scheme == 'xavier' or scheme == 'glorot':
        scale = (1/in_dim)**0.5
        W = np.random.normal(loc=0.0, scale=scale, size=(in_dim, out_dim))
        b = np.random.randn(out_dim) * 0.0001
    else:
        W = np.random.standard_normal((in_dim, out_dim))
        b = np.random.randn(out_dim) * 0.0001

    return W, b

# Step 6 - make_loss
def make_loss(kind='cross_entropy'):
    """Return a classification loss_fn(logits, labels) -> (loss, d_logits).

    Inputs to loss_fn:
        logits: (batch, C) float array of raw class scores
        labels: (batch,) int array of class indices in [0, C)
    Outputs:
        loss: Python float, mean scalar loss over the batch (finite)
        d_logits: (batch, C) gradient of loss w.r.t. logits (finite)
    Must pass gradient_check, be minimized by confident correct predictions,
    and stay finite under saturated logits.
    """
    # TODO: your approach here
    if kind == 'cross_entropy':
        def loss_fn(x, y):
            shifted = x - x.max(axis=-1, keepdims=True)
            logsumexp = np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
            loss = np.mean(logsumexp - shifted[np.arange(y.shape[0]), y])
            dlogits = np.exp(shifted - logsumexp)
            dlogits[np.arange(y.shape[0]), y] -= 1.0
            dlogits /= y.shape[0]
            return loss, dlogits

        return loss_fn
    else:
        def loss_fn(x, y):
            one_hot = np.zeros_like(x)
            one_hot[np.arange(y.shape[0]), y] = 1.0
            loss = ((x - one_hot)**2).mean()
            dlogits = 2/x.shape[0] * (x - one_hot)
            return loss, dlogits

        return loss_fn

# Step 7 - make_sequential
def make_sequential(layers):
    """Compose protocol-honoring layers into one sequential model.

    Inputs:
      layers: list of layer dicts, each with
        forward(x) -> (y, cache),
        backward(dout, cache) -> (dx, grads_dict),
        params: dict of ndarrays (possibly empty).

    Returns a dict with:
      forward(x) -> (y, caches)
        y: final activation after applying every layer in order
        caches: opaque structure needed by backward
      backward(dout, caches) -> (dx, grads_list)
        dx: gradient w.r.t. the original input x
        grads_list: list of length len(layers); grads_list[i] is the
          grads_dict from layers[i] ({} for param-free layers)
      params: aggregated live view of all layer params, length len(layers),
        same order as layers (so in-place updates affect the model)
    """
    # TODO: your approach here
    params = [layer['params'] for layer in layers]


    def forward(x):
        y = x
        caches = []
        for layer in layers:
            y, cache = layer['forward'](y)
            caches.append(cache)
        return y, caches


    def backward(dout, caches):
        dx = dout
        grads_list = [None] * len(layers)
        for i in range(len(layers) - 1, -1, -1):
            layer = layers[i]
            cache = caches[i]
            dx, grads = layer['backward'](dx, cache)
            grads_list[i] = grads
        return dx, grads_list


    return dict(
      forward=forward,
      backward=backward,
      params=params
    )

# Step 8 - forward_backward
def forward_backward(model, loss_fn, x, y):
    """Run one full forward-backward sweep on a batch.

    Inputs:
      model: sequential dict with 'forward', 'backward', 'params'
             model['forward'](x) -> (logits, caches)
             model['backward'](d_logits, caches) -> (dx, param_grads)
      loss_fn: callable (logits, y) -> (loss, d_logits)
      x: np.ndarray (batch, in_dim)
      y: np.ndarray (batch,) integer labels

    Returns:
      loss: float, scalar batch loss
      param_grads: nested np.ndarrays matching model['params'] layout
                   (gradients of loss w.r.t. every parameter)
    """
    # TODO: your approach here
    logits, caches = model['forward'](x)
    loss, d_logits = loss_fn(logits, y)
    _, param_grads = model['backward'](d_logits, caches)

    return loss, param_grads

# Step 9 - make_optimizer
def make_optimizer(params, lr=1e-2, kind='sgd'):
    """Build an optimizer that updates params in place.

    Inputs:
      params: arrays, possibly nested in lists/dicts (or dict of arrays) to optimize
      lr: float learning rate
      kind: str algorithm name (e.g. 'sgd')

    Returns:
      dict with key 'step'. step(grads) applies one in-place update
      using grads structured like params. Parameter shapes must stay
      unchanged. Repeated steps must reduce a simple convex objective
      within a modest fixed budget and keep values finite.
    """
    # TODO: your approach here
    if kind == 'sgd':
        def step(grads):
            for i in range(len(params)):
                if isinstance(params[i], dict):
                    for key in params[i]:
                        params[i][key] -= lr * grads[i][key]
                elif isinstance(params[i], np.ndarray):
                    params[i] -= lr * grads[i]

    else:
        def step(grads, adam_state=None, beta1=0.9, beta2=0.99, eps=1e-5):
            if adam_state == None:
                adam_state = {'m': [], 'v': [], 't': 0}
                for param in params:
                    m = {key:np.zeros_like(val) for key, val in param.items()}
                    v = {key:np.zeros_like(val) for key, val in param.items()}
                    adam_state['m'].append(m)
                    adam_state['v'].append(v)

            adam_state['t'] += 1

            for i in range(len(params)):
                for key in grads[i]:
                    adam_state['m'][i][key] = beta1*adam_state['m'][i][key] + (1-beta1)*grads[i][key]
                    adam_state['v'][i][key] = beta2*adam_state['v'][i][key] + (1-beta2)*grads[i][key]**2

                    m_hat = adam_state['m'][i][key]/(1-beta1**adam_state[t])
                    v_hat = adam_state['v'][i][key]/(1-beta2**adam_state[t])
                    params[i][key] -= lr * m_hat/(np.sqrt(v_hat)+eps)

    return {'step' : step}

# Step 10 - train_step
def train_step(model, loss_fn, optimizer, x_batch, y_batch):
    """Perform one complete optimization step over a minibatch.

    Inputs:
      model: sequential model dict with 'forward', 'backward', and 'params'
      loss_fn: callable (logits, y) -> (loss, d_logits)
      optimizer: dict with 'step'(grads) applying in-place parameter updates
      x_batch: np.ndarray of shape (B, D)
      y_batch: np.ndarray of shape (B,) integer class labels

    Returns:
      loss: float, scalar batch loss evaluated BEFORE the parameter update.
      Model parameters are updated in place; shapes unchanged and values finite.
    """
    # TODO: your approach here
    loss, param_grads = forward_backward(model, loss_fn, x_batch, y_batch)
    optimizer['step'](param_grads)

    return loss

# Step 11 - train (not yet solved)
# TODO: implement

# Step 12 - design_network (not yet solved)
# TODO: implement

# Step 13 - improve_generalization (not yet solved)
# TODO: implement

