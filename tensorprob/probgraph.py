from collections import namedtuple, OrderedDict
from itertools import product
import logging

import tensorflow as tf

from . import config
from .utilities import set_logp_to_neg_inf

logger = logging.getLogger('tensorprob')


# TODO Change name of Helper
# TODO Do we need a getter here?
Helper = namedtuple('Helper', ['bounds', 'setter', 'setter_var'])

Graph = namedtuple('Graph', ['logp', 'integral', 'rvs'])

Observable = namedtuple('Observable', ['dummy', 'setter', 'setter_var'])


class ProbGraph(object):
    def __init__(self):
        # Dictionary of the form {distribution: Helper} to keep track of the
        # distributions present in the model
        self._helpers = {}
        # Dictionary of the form {distributions: Graph} where distributions is
        # a tuple of distributions which are present in _helpers
        self._graphs = {}
        # TODO document
        self._useable_dists = []
        self._observed = None
        self._hidden = None

    def add_dist(self, dist, bounds, setter=None, setter_var=None):
        """Register a new `Helper` in the `ProbGraph`.

        # Arguments
            dist: tensorflow.Tensor
            bounds: list of Region objects
            setter: optional, tensorflow expression
                Argument to pass to `tensorflow.run` to run the assign a value
                to dist, defaults to a trival `dist.assign()` operation.
            setter:
                The variable to place in the `feed_dict` when executing `setter`.
        """
        assert (setter, setter_var) == (None, None) or None not in (setter, setter_var)

        if setter is None:
            setter_var = tf.Variable(dist.dtype.as_numpy_dtype(),
                                     name=dist.name.split(':')[0]+'_setter')
            setter = dist.assign(setter_var)

        self._helpers[dist] = Helper(bounds, setter, setter_var)
        self._useable_dists.append(dist)

    def add_graph(self, dists, logp, integral=None, rvs=None):
        """Register a new `Graph` in the `ProbGraph`.

        # Arguments
            dists: tensorflow.Tensor or tuple of tensorflow.Tensor
            logp: tensorflow expression
            integral: optional, tensorflow expression
            rvs: optional, function
        """
        assert integral is not None

        if not isinstance(dists, tuple):
            dists = tuple(dists)

        self._graphs[dists] = Graph(logp, integral, rvs)

    def finalize(self):
        """Finalise the ProbGraph.

            - Normalises all logps
            - Restricts logps to be -inf outside of the bounded regions
            - and probably does additional things in the future.
        """
        for dists, (logp, integral, rvs) in self._graphs.items():
            # Normalize the logp
            integrals = []
            for bounds in product(*[self._helpers[d].bounds for d in dists]):
                # Bounds is currently a list of regions for each dimension.
                # Convert it to instead be a list of lower bounds and a list
                # of upper bounds.
                integrals = integral(*zip(*bounds))
            logp -= tf.log(tf.add_n(integrals))

            # Set logp to -inf when outside each distribution's bounds
            for dist in dists:
                logp = set_logp_to_neg_inf(dist, logp, self._helpers[dist].bounds)

            # Update the graph to reflect these changes
            self._graphs[dists] = Graph(logp, integral, rvs)

    def set_observed(self, observed):
        # Raise an error if only part of a multidimensionl distribution is specified
        for graph_key in self._graphs:
            if sum(dist in observed for dist in graph_key) != len(graph_key):
                raise ValueError("Multidimensional distribution partially observed")

        self._observed = OrderedDict()
        for dist in observed:
            # Make a new assign operation to replace the placeholder
            dummy = self._helpers[dist].setters_var
            setter_var = tf.Variable(dist.dtype.as_numpy_dtype(), name=dist.name.split(':')[0])
            setter = tf.assign(dummy, setter_var, validate_shape=False)

            self._observed[dist] = Observable(dummy, setter, setter_var)

    def initialize(self, assign_dict):
        hidden = set(self._useable_dists).difference(set(self._observed))
        if hidden != set(assign_dict.keys()):
            # TODO Move all Exceptions to an exceptions module
            # raise ModelError("Not all latent variables have been passed in a "
            raise RuntimeError("Not all latent variables have been passed in a "
                               "call to `model.initialize().\nMissing variables: {}"
                               .format(hidden.difference(assign_dict.keys())))

        # Add variables to the execution graph
        self._hidden = OrderedDict()
        # Sort the hidden variables so we can access them in a consistant order
        for var in sorted(hidden, key=lambda v: v.name):
            self._hidden[var] = self._helpers[var].setter_var
        self.session.run(tf.initialize_variables(list(self._hidden.values())))

        for h in self._hidden.values():
            var = tf.Variable(h.dtype.as_numpy_dtype(),
                              name=h.name.split(':')[0] + '_placeholder')
            setter = h.assign(var)
            self._setters[h] = (setter, var)

        all_vars = self._hidden.copy()
        all_vars.update(self._observed)

        self._rewrite_graph(all_vars)

        # observed_logps contains one element per data point
        observed_logps = []
        hidden_logps = []
        for graph_key, graph in self._graph.items():
            if graph_key[0] in self._observed:
                observed_logps.append(self._get_rewritten(graph.logp))
            else:
                hidden_logps.append(self._get_rewritten(graph.logp))

        # Handle the case where we don't have observed variables.
        # We define the probability to not observe anything as 1.
        if not observed_logps:
            observed_logps = [tf.constant(0, dtype=config.dtype)]

        self._pdf = tf.exp(tf.add_n(observed_logps))

        self._nll = -tf.add_n(
            [tf.reduce_sum(logp) for logp in observed_logps] +
            hidden_logps
        )

        variables = list(self._hidden.values())
        self._nll_grad = tf.gradients(self._nll, variables)
        for i, (v, g) in enumerate(zip(variables, self._nll_grad)):
            if g is None:
                self._nll_grad[i] = tf.constant(0, dtype=config.dtype)
                logger.warn('Model is independent of variable {}'.format(
                    v.name.split(':')[0]
                ))

    def assign(self, values_dict):
        setters = []
        feed_dict = {}
        for dist, value in values_dict:
            setters.append(self._helpers[dist].setter)
            feed_dict[self._helpers[dist].setter_var] = value
        return setters, feed_dict

    def _rewrite_graph(self, transform):
        input_map = {k.name: v for k, v in transform.items()}

        # Modify the input dictionary to replace variables which have been
        # superseded with the use of combinators
        for k, v in self._silently_replace.items():
            input_map[k.name] = self._observed[v]

        try:
            tf.import_graph_def(
                    self._model_graph.as_graph_def(),
                    input_map=input_map,
                    name='added',
            )
        except ValueError:
            # Ignore errors that ocour when the input_map tries to
            # rewrite a variable that isn't present in the graph
            pass

    def _get_rewritten(self, tensor):
        return self.session.graph.get_tensor_by_name('added/' + tensor.name)
