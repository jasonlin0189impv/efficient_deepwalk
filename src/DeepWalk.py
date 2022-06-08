from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from src.RandomWalk import RandomWalker as random_walk
from src.ERandomWalk import RandomWalker as efficient_random_walk


class CallBack(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0
        self.loss_previous_step = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        loss_now = loss - self.loss_previous_step
        self.loss_previous_step = loss
        print(f"Loss after epoch {self.epoch}: {loss_now}")
        self.epoch += 1


class DeepWalk:
    def __init__(self, graph, method):
        self.graph = graph
        assert method in [
            "random_walk",
            "efficient_random_walk",
        ], "We only support 'random_walk' and 'efficient_random_walk'"
        self.method = method
        self.sentences = None
        self.w2v_model = None

    def random_walk(self, walk_len=10, num_walks=30, n_workers=2):
        """RandomWalk method

        Args:
            walk_len (int): Walking length.
            num_walks (int): Number of walks.
            n_workers (int): Number of workers.
        """
        if self.method == "random_walk":
            walker = random_walk(self.graph, num_walks)
        else:
            walker = efficient_random_walk(self.graph, num_walks)

        self.sentences = walker.multi_walk(walk_len, n_workers)

    def train(self, embed_size=128, window_size=5, epochs=5, **kwargs):
        kwargs["sentences"] = self.sentences
        kwargs["vector_size"] = embed_size
        kwargs["window"] = window_size
        kwargs["negative"] = kwargs.get("negative", 5)
        kwargs["epochs"] = epochs
        kwargs["workers"] = kwargs.get("workers", 3)
        kwargs["seed"] = kwargs.get("seed", 10)
        kwargs["alpha"] = kwargs.get("alpha", 0.025)
        kwargs["compute_loss"] = kwargs.get("compute_loss", True)
        kwargs["callbacks"] = kwargs.get("callbacks", [CallBack()])
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["sg"] = kwargs.get("sg", 1)

        model = Word2Vec(**kwargs)
        self.w2v_model = model

    def get_embeds(self):
        embeds = {}
        for node in self.graph.nodes():
            embeds[node] = self.w2v_model.wv[node]
        return embeds
