import os
import yaml
import torch
from transformers import AlbertConfig, AlbertModel

class CustomAlbert(AlbertModel):
    def forward(self, *args, **kwargs):
        # Call the original forward method
        outputs = super().forward(*args, **kwargs)

        # Only return the last_hidden_state
        return outputs.last_hidden_state


def load_plbert(log_dir, vocab_size=None):
    config_path = os.path.join(log_dir, "config.yml")
    plbert_config = yaml.safe_load(open(config_path))
    if vocab_size is not None:
        plbert_config['model_params']['vocab_size'] = vocab_size
    
    albert_base_configuration = AlbertConfig(**plbert_config['model_params'])
    # Prefer PyTorch native scaled dot-product attention for lower memory use.
    # Transformers may default to SDPA for ALBERT already, but making this explicit
    # keeps both training stages deterministic across Transformers versions.
    try:
        bert = CustomAlbert._from_config(albert_base_configuration, attn_implementation='sdpa')
    except Exception:
        bert = CustomAlbert(albert_base_configuration)

    files = os.listdir(log_dir)
    ckpts = []
    for f in os.listdir(log_dir):
        if f.startswith("step_"): ckpts.append(f)

    iters = [int(f.split('_')[-1].split('.')[0]) for f in ckpts if os.path.isfile(os.path.join(log_dir, f))]
    iters = sorted(iters)[-1]

    checkpoint = torch.load(log_dir + "/step_" + str(iters) + ".t7", map_location='cpu')
    state_dict = checkpoint['net']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        if name.startswith('encoder.'):
            name = name[8:] # remove `encoder.`
            new_state_dict[name] = v
    del new_state_dict["embeddings.position_ids"]

    word_key = 'embeddings.word_embeddings.weight'
    if word_key in new_state_dict and new_state_dict[word_key].shape != bert.state_dict()[word_key].shape:
        old_weight = new_state_dict[word_key]
        new_weight = bert.state_dict()[word_key].clone()
        rows = min(old_weight.shape[0], new_weight.shape[0])
        new_weight[:rows] = old_weight[:rows]
        if new_weight.shape[0] > old_weight.shape[0]:
            new_weight[rows:] = old_weight.mean(dim=0, keepdim=True)
        new_state_dict[word_key] = new_weight

    bert.load_state_dict(new_state_dict, strict=False)
    
    return bert
