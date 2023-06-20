import os
import torch
import numpy as np
import bits_back.util as util
import bits_back.rans as rans
from bits_back.torch_vae.tvae_vanilla import BetaBinomialVAE
from bits_back.torch_vae import tvae_utils
import time
import sar

rng = np.random.RandomState(0)
np.seterr(all='raise')

randomise_data = True

prior_precision = 8
gaussian_precision = 12
q_precision = 14

compress_lengths = []

latent_shape = (1,16)
model = BetaBinomialVAE()
model.load_state_dict(torch.load('results/torch_vae_params_obgn_arxiv'))
model.eval()

rec_net = tvae_utils.torch_fun_to_numpy_fun(model.encode)
gen_net = tvae_utils.torch_fun_to_numpy_fun(model.decode)

obs_append = tvae_utils.beta_binomial_obs_append(gaussian_precision)
obs_pop = tvae_utils.beta_binomail_obs_pop(gaussian_precision)

vae_append = util.vae_append(latent_shape, gen_net, rec_net, obs_append,
                             prior_precision, q_precision)
vae_pop = util.vae_pop(latent_shape, gen_net, rec_net, obs_pop,
                       prior_precision, q_precision)

if __name__ == '__main__':
    partitioning_json_file = "./partition_data_2/ogbn-arxiv.json"
    rank = 0
    device = "cpu"

    # Load DGL partition data
    partition_data = sar.load_dgl_partition_data(partitioning_json_file, rank, False, device)
    masks = {}
    for mask_name, indices_name in zip(['train_mask', 'val_mask', 'test_mask'],
                    ['train_indices', 'val_indices', 'test_indices']):
        boolean_mask = sar.suffix_key_lookup(partition_data.node_features, mask_name)
        masks[indices_name] = boolean_mask.nonzero(as_tuple=False).view(-1).to(device)
    labels = sar.suffix_key_lookup(partition_data.node_features, 'labels').long().to(device)
    features = sar.suffix_key_lookup(partition_data.node_features, 'features').to(device)
    features = features[masks["test_indices"]]

    # randomly generate some 'other' bits
    other_bits = rng.randint(low=1 << 16, high=1 << 31, size=20, dtype=np.uint32)
    state = rans.unflatten(other_bits)

    print_interval = 10
    encode_start_time = time.time()
    for i, feat in enumerate(features):
        feat = feat.unsqueeze(0)
        state = vae_append(state, feat)

        if not i % print_interval:
            print('Encoded {}'.format(i))

        compressed_length = 32*(len(rans.flatten(state)) - len(other_bits)) / (i+1)
        compress_lengths.append(compressed_length)

    print('\nAll encoded in {:.2f}s'.format(time.time() - encode_start_time))
    compressed_message = rans.flatten(state)

    compressed_bits = 32 * (len(compressed_message) - len(other_bits))
    print("Used " + str(compressed_bits) + " bits.")
    print('This is {:.2f} bits per pixel'.format(compressed_bits / (len(features)*128.)))

    state = rans.unflatten(compressed_message)
    decode_start_time = time.time()

    for n in range(len(features)):
        state, feat_ = vae_pop(state)
        original_feat = features[len(features)-n-1].numpy()
        assert all(original_feat == np.array(feat_))

        if not n % print_interval:
            print('Decoded {}'.format(n))

    print('\nAll decoded in {:.2f}s'.format(time.time() - decode_start_time))

    recovered_bits = rans.flatten(state)
    assert all(other_bits == recovered_bits)
