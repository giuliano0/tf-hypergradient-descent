import tensorflow as tf

# FIXME: When tf2.11 arrives, experimental will go live
from tensorflow.keras.optimizers import experimental as opt

class HyperAdam(opt.Optimizer):
    def __init__(
            self,
            learning_rate: float = 0.001,
            beta_1: float = 0.9,
            beta_2: float = 0.999,
            beta_h: float = 0.1,
            epsilon: float = 1e-7,
            amsgrad: bool = False,
            name: str = 'HyperAdam',
            **kwargs
        ):
        """REEEEEEEEEEEEEEEEEEEEEE"""

        super().__init__(name, **kwargs)

        # NOTE: self.iterations exists, no need to create it

        # FIXME: Remove when/if I implement amsgrad
        # (not sure if it's compatible with hypergrad)
        if amsgrad:
            raise NotImplementedError('Sorry, too lazy.')

        # General hyperparams
        self.initial_learning_rate = learning_rate
        self._learning_rate = self._build_learning_rate(learning_rate)

        # Adam hyperparams
        self.b1 = beta_1
        self.b2 = beta_2

        # Hypergrad hyperparams
        self.bh = beta_h

        # Extras
        self.eps = epsilon
        self.amsgrad = amsgrad # Iff hypergrad is compatible (spoiler: doesn't seem likely)

    def build(self, var_list):
        super().build(var_list)

        if hasattr(self, "_built") and self._built:
            return

        self._built = True

        self.m1 = []
        self.m2 = []
        self.du = []

        self.alpha = self.add_variable_from_reference(
            model_variable=self.lr,
            variable_name='alpha'
        )

        for var in var_list:
            # self.add_variable adds a variable that I shape, while
            # self.add_variable_from_reference takes variable to shape the optimizer variable for me
            self.m1.append(
                self.add_variable_from_reference(model_variable=var, variable_name='m')
            )
            self.m2.append(
                self.add_variable_from_reference(model_variable=var, variable_name='v')
            )
            self.du.append(
                self.add_variable_from_reference(model_variable=var, variable_name='du')
            )

        if self.amsgrad:
            # Nothing to do here
            pass

    def update_step(self, gradient, variable):
        beta_1_power = tf.cast(self.b1, variable.dtype)
        beta_2_power = tf.cast(self.b2, variable.dtype)

        # This is the local iteration, starting at 1
        it = tf.cast(self.iterations + 1, variable.dtype)

        # NOTE: My tests shown tf.pow can be more stable than
        # iterative multiplication, specially for lower expoents
        beta_1_power = tf.pow(beta_1_power, it)
        beta_2_power = tf.pow(beta_2_power, it)

        # This factors out the betas to create alpha_t = beta_stuff_t * alpha
        beta_factor = tf.sqrt(1 - beta_2_power) / (1 - beta_1_power)

        if isinstance(gradient, tf.IndexedSlices):
            raise NotImplementedError('Sorry, too lazy.')

        var_key = self._var_key(variable)
        m = self.m1[self._index_dict[var_key]]
        v = self.m2[self._index_dict[var_key]]
        du = self.du[self._index_dict[var_key]]

        # Learning rate update step
        # NOTE: du doesn't carry the minus signal
        # so this is alpha = alpha + beta_h * (g * du)
        # NOTE: Couldn't manage to use tf.tensordot because the rank of
        # gradient wasn't available, but the line below is equivalent
        self.alpha.assign_add(
            self.bh * tf.reduce_sum(gradient * du)
        )

        m.assign_add((gradient - m) * (1 - self.b1))
        v.assign_add((tf.square(gradient) - v) * (1 - self.b2))

        du.assign(
            beta_factor * m / (tf.sqrt(v) + self.eps)
        )

        # NOTE: du doesn't carry the minus signal
        # so this is var = var - alpha * du
        variable.assign_sub(
            self.alpha * du
        )

    def get_config(self):
        config = super().get_config()

        config.update({
            'learning_rate': self.lr,
            'beta_1': self.b1,
            'beta_2': self.b2,
            'beta_h': self.bh,
            'epsilon': self.eps,
            'amsgrad': self.amsgrad
        })

        return config
