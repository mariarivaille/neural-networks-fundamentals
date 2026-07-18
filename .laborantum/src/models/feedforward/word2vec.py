import torch
import torch.nn.functional as F


class BinaryIndexTree:
    def __init__(self, vocab_size):
        self.vocab_size = int(vocab_size)
        if self.vocab_size <= 0:
            raise ValueError('vocab_size must be positive')
        self.depth = max(1, (self.vocab_size - 1).bit_length())
        self.max_path_length = self.depth
        self.num_internal_nodes = 2 ** self.depth - 1

    def targets_for_index(self, word_index):
        word_index = int(word_index)
        if word_index < 0 or word_index >= self.vocab_size:
            raise ValueError(f'word index {word_index} is outside vocabulary')
        return format(word_index, f'0{self.depth}b')

    @staticmethod
    def node_id_from_prefix(prefix_bits):
        if isinstance(prefix_bits, str):
            prefix = prefix_bits
        else:
            prefix = ''.join(str(int(bit)) for bit in prefix_bits)
        if not prefix:
            return 0
        depth = len(prefix)
        return (2 ** depth - 1) + int(prefix, 2)

    def path_and_targets(self, word_index):
        targets_string = self.targets_for_index(word_index)
        path = [
            self.node_id_from_prefix(targets_string[:step])
            for step in range(self.depth)
        ]
        targets = [int(bit) for bit in targets_string]
        return path, targets

    def __call__(self, context_word):
        device = context_word.device
        context_word = context_word.detach().cpu().view(-1).tolist()
        batch_size = len(context_word)

        paths = []
        targets_list = []
        for word_index in context_word:
            path, targets = self.path_and_targets(int(word_index))
            paths.append(path)
            targets_list.append(targets)

        return {
            'path': torch.tensor(paths, dtype=torch.long, device=device),
            'targets': torch.tensor(targets_list, dtype=torch.float32, device=device),
            'mask': torch.ones(batch_size, self.max_path_length, dtype=torch.float32, device=device),
        }


class HierarchicalSoftmax(torch.nn.Module):
    def __init__(self, embedding_dim, vocab_size):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.targets = BinaryIndexTree(vocab_size)
        self.decoder = torch.nn.Embedding(self.targets.num_internal_nodes, self.embedding_dim)
        self.decoder = torch.nn.Embedding(self.targets.num_internal_nodes, self.embedding_dim)
        torch.nn.init.normal_(self.decoder.weight, mean=0.0, std=0.02)

    @property
    def num_internal_nodes(self):
        return self.targets.num_internal_nodes

    @property
    def max_path_length(self):
        return self.targets.max_path_length

    def forward(self, embedding, target_word):
        target_tensors = self.targets(target_word)
        path = target_tensors['path']
        targets = target_tensors['targets']
        mask = target_tensors['mask']

        node_vectors = self.decoder(path)
        logits = torch.einsum('bd,bld->bl', embedding, node_vectors)
        probabilities = torch.sigmoid(logits)
        target_probabilities = torch.where(
            targets.bool(),
            probabilities,
            1.0 - probabilities,
        )
        total_probability = target_probabilities.prod(dim=1)
        per_node_loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction='none',
        )
        per_word_loss = (per_node_loss * mask).sum(dim=1)
        loss = per_word_loss.mean()

        return {
            **target_tensors,
            'logits': logits,
            'probabilities': probabilities,
            'target_probabilities': target_probabilities,
            'total_probability': total_probability,
            'per_node_loss': per_node_loss,
            'per_word_loss': per_word_loss,
            'loss': loss,
        }


class Word2Vec(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.embedding_dim = int(embedding_dim)
        self.encoder = torch.nn.Embedding(self.vocab_size, self.embedding_dim)
        self.hierarchical_softmax = HierarchicalSoftmax(self.embedding_dim, self.vocab_size)
        self.encoder = torch.nn.Embedding(self.vocab_size, self.embedding_dim)
        self.hierarchical_softmax = HierarchicalSoftmax(self.embedding_dim, self.vocab_size)
        self.decoder = self.hierarchical_softmax.decoder
        self.num_internal_nodes = self.hierarchical_softmax.num_internal_nodes
        torch.nn.init.normal_(self.encoder.weight, mean=0.0, std=0.02)

    def forward(self, batch):
        center_word = batch['data']['center_word']
        embedding = self.encoder(center_word)
        batch['signals'] = {'embedding': embedding}
        batch['postprocessed'] = {}
        if 'context_word' in batch['data']:
            out = self.hierarchical_softmax(embedding, batch['data']['context_word'])
            batch['data']['path'] = out['path']
            batch['data']['targets'] = out['targets']
            batch['data']['mask'] = out['mask']
            batch['signals']['logits'] = out['logits']
            batch['signals']['probabilities'] = out['probabilities']
            batch['signals']['target_probabilities'] = out['target_probabilities']
            batch['signals']['total_probability'] = out['total_probability']
            batch['signals']['loss'] = out['loss']
            batch['postprocessed']['targets'] = (out['probabilities'] >= 0.5).long()
        return batch
