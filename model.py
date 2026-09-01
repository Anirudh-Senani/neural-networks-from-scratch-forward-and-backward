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
def make_optimizer(params, lr=1e-2, kind='sgd', l2_lambda=None):
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
                        if l2_lambda is not None:
                            params[i][key] -= lr * l2_lambda * params[i][key]
                        params[i][key] -= lr * grads[i][key]
                elif isinstance(params[i], np.ndarray):
                    if l2_lambda is not None:
                        params[i] -= lr * l2_lambda * params[i]
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

# Step 11 - train
def train(model, loss_fn, optimizer, x, y, epochs, batch_size, seed=0, early_stopping=None):
    """Run a deterministic minibatch training loop.

    Inputs:
      model: sequential model dict with 'forward', 'backward', 'params'
      loss_fn: callable (logits, y) -> (loss, d_logits)
      optimizer: dict with 'step'(grads) applying in-place parameter updates
      x: np.ndarray of shape (N, D) training features
      y: np.ndarray of shape (N,) integer class labels
      epochs: int, number of full passes over the data
      batch_size: int, minibatch size
      seed: int, RNG seed for deterministic shuffling / batching

    Returns:
      history: list[float] of length `epochs`; history[t] is the mean
      train_step loss over minibatches in epoch t.
      Model parameters are updated in place; shapes unchanged.
    """
    # TODO: your approach here
    rng = np.random.default_rng(seed)
    num_samples = y.shape[0]
    inds = np.arange(num_samples)
    rng.shuffle(inds)
    history = []
    patience = 0

    for ep in range(epochs):
        loss = 0.0
        for i in range(0,num_samples,batch_size):
            x_batch = x[inds[i:i+batch_size], :]
            y_batch = y[inds[i:i+batch_size]]

            batch_loss = train_step(model, loss_fn, optimizer, x_batch, y_batch)
            loss += batch_loss * min(batch_size, y_batch.shape[0])

        history.append(loss/num_samples)

        if early_stopping is not None and ep > 2:
            if history[-2] - history[-1] < early_stopping:
                patience += 1
            else:
                patience = 0

            if patience >= 2:
                break

    return history

# Step 12 - design_network
def generate_pie_clusters(num_clusters=4, points_per_cluster=250, radial_noise=0.1):
    """
    Generates concentric ring clusters (linearly inseparable).
    Each cluster occupies a radial band and spans the full 360 degrees,
    so no single straight line can separate the classes.
    """
    total_points = num_clusters * points_per_cluster
    labels = np.repeat(np.arange(num_clusters), points_per_cluster)

    # full angular range for every cluster (no angular separation anymore)
    theta = np.random.rand(total_points) * 2 * np.pi

    # radius now encodes the class -> concentric rings
    base_radius = labels + 1  # ring 1, 2, 3, ... for each class
    radius = base_radius + np.random.normal(0, radial_noise, total_points)
    radius = np.clip(radius, 0, None)

    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    features = np.column_stack((x, y))
    return features, labels


def design_network(input_dim, num_classes, seed=0):
    """Design and train a net that solves a nonlinear classification task.

    Inputs:
        input_dim: int, feature dimension
        num_classes: int, number of classes
        seed: int, RNG seed for reproducibility

    Returns:
        model: trained sequential model (forward/backward/params)
        metrics: dict with
            'accuracy': float >= 0.90 on an evaluation set,
            'x': np.ndarray (N, input_dim) eval features (N >= 50),
            'y': np.ndarray (N,) integer eval labels.
        The eval set (x, y) must not be linearly separable to high accuracy
        (< 0.82 for a linear classifier), and the model's true accuracy on
        it must match metrics['accuracy'] and be >= 0.90.
    """
    # TODO: your approach here
    # rng = np.random.default_rng(seed)
    np.random.seed(seed)
    x, y = generate_pie_clusters(
        num_clusters=num_classes, points_per_cluster=512, radial_noise=0.05)
    # x, y = generate_concentric_rings(n_samples=1280, label_noise=0.05)

    if input_dim > 2:
        x_in = np.random.rand(x.shape[0], input_dim) * 0.02
        x_in[:, :2] = x
    else:
        x_in = x

    in_dim = input_dim
    hidden = 32
    # hidden1 = in_dim*32
    # hidden2 = in_dim*16
    # hidden3 = in_dim*8
    # hidden3 = num_classes*32
    out_dim = num_classes

    epochs = 50
    batch_size = 64
    eval_size = 128

    layers = [
        make_dense(in_dim, hidden, initialize_weights),
        make_activation(),
        make_dense(hidden, hidden, initialize_weights),
        make_activation(),
        make_dense(hidden, out_dim, initialize_weights)
    ]

    inds = np.arange(x_in.shape[0])
    np.random.shuffle(inds)
    x = x_in[inds]
    y = y[inds]

    x_train = x[:-eval_size]
    y_train = y[:-eval_size]
    x_eval = x[-eval_size:]
    y_eval = y[-eval_size:]

    model = make_sequential(layers)
    loss_fn = make_loss()
    optimizer = make_optimizer(model['params'])
    history = train(model, loss_fn, optimizer, x_train,
                    y_train, epochs, batch_size, seed)

    logits, _ = model['forward'](x_eval)
    eval_pred = np.argmax(logits, axis=-1)
    accuracy = (eval_pred == y_eval).sum()/eval_size

    return model, dict(accuracy=accuracy, x=x_eval, y=y_eval)

# Step 13 - improve_generalization
def improve_generalization(baseline_model_fn, x_train, y_train, x_val, y_val, seed=0):
    """Improve held-out accuracy over an unregularized baseline.

    Inputs:
        baseline_model_fn: zero-arg callable -> fresh untrained sequential model
            (dict with 'forward', 'backward', 'params') matching the data dims.
        x_train, y_train: training features (N, D) and int labels (N,).
        x_val, y_val: validation features (N_val, D) and int labels (N_val,).
        seed: int for deterministic training.

    Returns:
        dict with keys:
            'val_accuracy': float accuracy of the improved model on x_val/y_val
            'baseline_val_accuracy': float val accuracy of plain unregularized SGD
            'predictions': np.ndarray shape (N_val,) int preds from improved model
            'model': the trained improved model

    Required behavior:
        val_accuracy > baseline_val_accuracy
        predictions == argmax(model.forward(x_val), axis=1)
        val_accuracy == mean(predictions == y_val)
        predictions are non-constant (not a trivial single-class predictor)
    """
    # TODO: your approach here
    np.random.seed(seed)
    baseline_model = baseline_model_fn()
    loss_fn = make_loss()
    baseline_optimizer = make_optimizer(baseline_model['params'])

    epochs = 50
    batch_size = 64
    early_stopping = 0.001
    l2_lambda = 1e-3
    _ = train(baseline_model, loss_fn, baseline_optimizer, x_train, y_train, epochs, batch_size, seed)

    baseline_preds = np.argmax(baseline_model['forward'](x_val)[0], axis=-1)
    baseline_val_accuracy = np.mean(baseline_preds == y_val)

    model = baseline_model_fn()
    optimizer = make_optimizer(model['params'], l2_lambda=l2_lambda)
    history = train(model, loss_fn, optimizer, x_train, y_train, epochs, batch_size, seed, early_stopping)

    preds = np.argmax(model['forward'](x_val)[0], axis=-1)
    val_accuracy = np.mean(preds == y_val)

    return dict(
        baseline_val_accuracy=baseline_val_accuracy,
        val_accuracy=val_accuracy,
        predictions=preds,
        model=model
    )

