class Configuration(object):
    """
    The configuration instance list the hyperparameters of
    the model, inspired from [deepmind/sonnet].

    References:
        [deepmind/sonnet] https://github.com/deepmind/sonnet/blob/master/sonnet/examples/vqvae_example.ipynb.

        [van den Oord et al., 2017] van den Oord A., and Oriol Vinyals. "Neural discrete representation
        learning." Advances in Neural Information Processing Systems(NIPS). 2017.

        [Roy et al., 2018] A. Roy, A. Vaswani, A. Neelakantan, and N. Parmar. Theory and experiments on vector
        quantized autoencoders.arXiv preprint arXiv:1805.11063, 2018.
    """

    def __init__(self, batch_size=32, num_training_updates=25000, \
        num_hiddens=128, num_residual_hiddens=32, num_residual_layers=2, \
        embedding_dim=64, num_embeddings=512, commitment_cost=0.25, \
        decay=0.99, learning_rate=3e-4):

        self._batch_size = batch_size
        self._num_training_updates = num_training_updates
        self._num_hiddens = num_hiddens
        self._num_residual_hiddens = num_residual_hiddens
        self._num_residual_layers = num_residual_layers

        """
        This value is not that important, usually 64 works.
        This will not change the capacity in the information-bottleneck.
        """
        self._embedding_dim = embedding_dim

        # The higher this value, the higher the capacity in the information bottleneck.
        self._num_embeddings = num_embeddings

        """
        Commitment cost should be set appropriately. It's often useful to try a couple
        of values. It mostly depends on the scale of the reconstruction cost
        (log p(x|z)). So if the reconstruction cost is 100x higher, the
        commitment_cost should also be multiplied with the same amount.
        """
        self._commitment_cost = commitment_cost

        """
        Only uses for the EMA updates (instead of the Adam optimizer).
        This typically converges faster, and makes the model less dependent on choice
        of the optimizer. In the original VQ-VAE paper [van den Oord et al., 2017],
        EMA updates were not used (but was developed afterwards) on [Roy et al., 2018].
        """
        self._decay = decay

        self._learning_rate = learning_rate

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def num_training_updates(self):
        return self._num_training_updates

    @property
    def num_hiddens(self):
        return self._num_hiddens

    @property
    def num_residual_hiddens(self):
        return self._num_residual_hiddens

    @property
    def num_residual_layers(self):
        return self._num_residual_layers

    @property
    def embedding_dim(self):
        return self._embedding_dim

    @property
    def num_embeddings(self):
        return self._num_embeddings

    @property
    def commitment_cost(self):
        return self._commitment_cost

    @property
    def decay(self):
        return self._decay

    @property
    def learning_rate(self):
        return self._learning_rate
