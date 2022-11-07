# Hypergradient Descent on tensorflow
Paper: [Online Learning Rate Adaptation with Hypergradient Descent](https://arxiv.org/abs/1703.04782)

Tensorflow implementation. Made for Tensorflow's new Optimiser API, to debut with version 2.11.

That means it imports the base class resolving the following import:

```python
from tensorflow.keras.optimizers.experimental import Optimizer
```

But when v2.11 comes, it'll have to point as follows:

```python
from tensorflow.keras.optimizers import Optimizer
```

This version currently implements the **additive** version of the learning rate update. There's a **multiplicative** version described by the authors but I didn't meddle with it yet.

The multiplicative version is deemed *faster*, but as I'm more concerned with the convergence of Adam than on it's speed, for now. I'll shelf that one for a little while.

**But where are the tests?** \
I tested it. Just barely. Trust.
