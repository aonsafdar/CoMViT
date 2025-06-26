import numpy as np

class TransformerClassifier(Module):
    def __init__(self,
                 seq_pool=True,
                 embedding_dim=768,
                 num_layers=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 num_classes=1000,
                 dropout=0.1,
                 attention_dropout=0.1,
                 stochastic_depth=0.1,
                 positional_embedding='learnable',
                 sequence_length=None,
                 use_orthogonal_head=False):
        super().__init__()
        # [existing code unchanged up to self.norm]

        self.norm = LayerNorm(embedding_dim)

        if use_orthogonal_head:
            # Generate Hadamard matrix as Kasami-like codebook
            n = 1
            while n < embedding_dim:
                n *= 2
            H = np.array([[1]])
            while H.shape[0] < n:
                H = np.block([[H, H],
                              [H, -H]])
            codebook = torch.tensor(H[:num_classes], dtype=torch.float32)
            self.codebook = nn.Parameter(codebook, requires_grad=False)
            self.fc = lambda x: F.linear(x, self.codebook)
        else:
            self.fc = Linear(embedding_dim, num_classes)

        self.apply(self.init_weight)

    # [rest of class unchanged]