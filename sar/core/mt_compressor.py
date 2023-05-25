

import pickle
import numpy as np
import torch
from torch.utils.cpp_extension import load
import time

t1 = time.time()
mt_compress = load(name="mt_compress", sources=["compression/mt_compress.cpp"], extra_cflags=['-fopenmp'])
print("Loaded mt_compress", time.time()-t1)
print()

ax = 1
results = []

for j in range(1, 100):
	print("N Partitions:", j)
	curr_time = 0.0
	curr_rate = 0.0
	for k in range(10):
		# tensors_l = [torch.rand(50,100000)]
		tensors_l = [torch.ones(50,100000)]
		# tensors_l = [torch.reshape(torch.arange(5000000),(50,100000))]
		# tensors_l = [torch.normal(0,1,size=(50,100000))]
		# print("Partitioning")
		t1 = time.time()
		partitioned_tensors = [torch.tensor_split(t, j,dim=ax) for t in tensors_l]
		partitioned_tensors = sum(partitioned_tensors,())
		partition_time = time.time() - t1
		pre_size = [t.element_size() * t.nelement() for t in partitioned_tensors]
		# print(time.time()-t1)
		# print("Compressing")
		t1 = time.time()
		cpp_vector = [pickle.dumps(t) for t in partitioned_tensors]
		# cpp_vector = mt_compress.byte_vector([pickle.dumps(t) for t in partitioned_tensors])
		compressed_tensors_l = mt_compress.mt_compress(cpp_vector)
		compressed_tensors_l = [torch.tensor(np.frombuffer(t, dtype=np.uint8)) for t in compressed_tensors_l]
		compress_time = time.time() - t1
		post_size = [t.element_size() * t.nelement() for t in compressed_tensors_l]
		# print(time.time()-t1)
		# print("Decompressing")
		t1 = time.time()
		compressed_tensors_l = [f.numpy().tobytes() for f in compressed_tensors_l]
		decompressed_tensors_l = mt_compress.mt_decompress(compressed_tensors_l)
		decompressed_tensors_l = [pickle.loads(t) for t in decompressed_tensors_l]
		decompress_time = time.time() - t1
		# print(time.time()-t1)
		# print("Reaggregating")
		t1 = time.time()
		decompressed_tensors_l = [torch.cat(decompressed_tensors_l[i:i+j],dim=ax) for i in range(int(len(decompressed_tensors_l)/j))]
		reagg_time = time.time() - t1
		# print(time.time()-t1) 
		# print()
		curr_time += (partition_time + compress_time + decompress_time + reagg_time)
		curr_rate += sum([s1 / s2 for s1, s2 in zip(pre_size, post_size)])
	curr_time = curr_time / 10.0
	curr_rate = curr_rate / 10.0
	print(curr_time, curr_rate)
	results.append((curr_time, curr_rate))

np.save("mt_compression_threads.npy", results)