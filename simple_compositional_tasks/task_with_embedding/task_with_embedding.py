class TaskWithEmbedding:
    def __init__(self, task, embedding, cfg):
        super().__init__()
        self.task = task
        self.embedding = embedding
