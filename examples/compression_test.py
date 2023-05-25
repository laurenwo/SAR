import torch
import numpy as np
import time
from multiprocessing import Pool

import pickle
import bz2

from sar.core.compressor import PickleCompressorDecompressor, Bz2CompressorDecompressor
from sar.core.compressor import LZMACompressorDecompressor, MT_GZipCompressorDecompressor

if __name__ == '__main__':
	random_small = torch.rand(1,1000) ## index 0
	random_med = torch.rand(10,10000)  ## index 1
	random_large = torch.rand(100,100000)  ## index 2

	ones_small = torch.ones(1,1000)  ## index 3
	ones_med = torch.ones(10,10000)  ## index 4
	ones_large = torch.ones(100,100000)  ## index 5

	arange_small = torch.reshape(torch.arange(1000), (1,1000))  ## index 6
	arange_med = torch.reshape(torch.arange(100000), (10, 10000))   ## index 7
	arange_large = torch.reshape(torch.arange(10000000),(100,100000))   ## index 8

	pickle_compressor = PickleCompressorDecompressor()  ## index 0
	bz2_compressor = Bz2CompressorDecompressor()  ## index 1
	lzma_compressor = LZMACompressorDecompressor()   ## index 2
	mt_gzip_compressor = MT_GZipCompressorDecompressor() 

	## compression method x tensor type x [de]compression time
	results = np.zeros((4,9,2))

	# ## random small
	# t1 = time.time()
	# random_small_compressed = pickle_compressor.compress([random_small])
	# t2 = time.time()
	# _ = pickle_compressor.decompress(random_small_compressed)
	# t3 = time.time()
	# results[0,0,0] = t2-t1
	# results[0,0,1] = t3-t2 
	# print("random small, pickle", t2-t1, t3-t2)

	# t1 = time.time()
	# random_small_compressed = bz2_compressor.compress([random_small])
	# t2 = time.time()
	# _ = bz2_compressor.decompress(random_small_compressed)
	# t3 = time.time()
	# results[1,0,0] = t2-t1
	# results[1,0,1] = t3-t2 
	# print("random small, bz2", t2-t1, t3-t2)

	# t1 = time.time()
	# random_small_compressed = lzma_compressor.compress([random_small])
	# t2 = time.time()
	# _ = lzma_compressor.decompress(random_small_compressed)
	# t3 = time.time()
	# results[2,0,0] = t2-t1
	# results[2,0,1] = t3-t2 
	# print("random small, lzma", t2-t1, t3-t2)

	# t1 = time.time()
	# random_small_compressed = mt_gzip_compressor.compress([random_small])
	# print("compress")
	# t2 = time.time()
	# _ = mt_gzip_compressor.decompress(random_small_compressed)
	# print("decompress")
	# t3 = time.time()
	# results[3,0,0] = t2-t1
	# results[3,0,1] = t3-t2 
	# print("random small, mt_gzip", t2-t1, t3-t2)

	# ## random med
	# t1 = time.time()
	# random_med_compressed = pickle_compressor.compress([random_med])
	# t2 = time.time()
	# _ = pickle_compressor.decompress(random_med_compressed)
	# t3 = time.time()
	# results[0,1,0] = t2-t1
	# results[0,1,1] = t3-t2 
	# print("random med, pickle", t2-t1, t3-t2)

	# t1 = time.time()
	# random_med_compressed = bz2_compressor.compress([random_med])
	# t2 = time.time()
	# _ = bz2_compressor.decompress(random_med_compressed)
	# t3 = time.time()
	# results[1,1,0] = t2-t1
	# results[1,1,1] = t3-t2 
	# print("random med, bz2", t2-t1, t3-t2)

	# t1 = time.time()
	# random_med_compressed = lzma_compressor.compress([random_med])
	# t2 = time.time()
	# _ = lzma_compressor.decompress(random_med_compressed)
	# t3 = time.time()
	# results[2,1,0] = t2-t1
	# results[2,1,1] = t3-t2 
	# print("random med, lzma", t2-t1, t3-t2)

	# t1 = time.time()
	# random_med_compressed = mt_gzip_compressor.compress([random_med])
	# print("compress")
	# t2 = time.time()
	# _ = mt_gzip_compressor.decompress(random_med_compressed)
	# print("decompress")
	# t3 = time.time()
	# results[3,1,0] = t2-t1
	# results[3,1,1] = t3-t2 
	# print("random med, mt_gzip", t2-t1, t3-t2)

	## random large
	t1 = time.time()
	random_large_compressed = pickle_compressor.compress([random_large])
	t2 = time.time()
	_ = pickle_compressor.decompress(random_large_compressed)
	t3 = time.time()
	results[0,2,0] = t2-t1
	results[0,2,1] = t3-t2 
	print("random large, pickle", t2-t1, t3-t2)

	t1 = time.time()
	random_large_compressed = bz2_compressor.compress([random_large])
	t2 = time.time()
	_ = bz2_compressor.decompress(random_large_compressed)
	t3 = time.time()
	results[1,2,0] = t2-t1
	results[1,2,1] = t3-t2 
	print("random large, bz2", t2-t1, t3-t2)

	t1 = time.time()
	random_large_compressed = lzma_compressor.compress([random_large])
	t2 = time.time()
	_ = lzma_compressor.decompress(random_large_compressed)
	t3 = time.time()
	results[2,2,0] = t2-t1
	results[2,2,1] = t3-t2 
	print("random large, lzma", t2-t1, t3-t2)

	t1 = time.time()
	random_large_compressed = mt_gzip_compressor.compress([random_large])
	print("compress")
	t2 = time.time()
	_ = mt_gzip_compressor.decompress(random_large_compressed)
	print("decompress")
	t3 = time.time()
	results[3,2,0] = t2-t1
	results[3,2,1] = t3-t2 
	print("random large, mt_gzip", t2-t1, t3-t2)

	# ## ones small
	# t1 = time.time()
	# ones_small_compressed = pickle_compressor.compress([ones_small])
	# t2 = time.time()
	# _ = pickle_compressor.decompress(ones_small_compressed)
	# t3 = time.time()
	# results[0,3,0] = t2-t1
	# results[0,3,1] = t3-t2 
	# print("ones small, pickle", t2-t1, t3-t2)

	# t1 = time.time()
	# ones_small_compressed = bz2_compressor.compress([ones_small])
	# t2 = time.time()
	# _ = bz2_compressor.decompress(ones_small_compressed)
	# t3 = time.time()
	# results[1,3,0] = t2-t1
	# results[1,3,1] = t3-t2 
	# print("ones small, bz2", t2-t1, t3-t2)

	# t1 = time.time()
	# ones_small_compressed = lzma_compressor.compress([ones_small])
	# t2 = time.time()
	# _ = lzma_compressor.decompress(ones_small_compressed)
	# t3 = time.time()
	# results[2,3,0] = t2-t1
	# results[2,3,1] = t3-t2 
	# print("ones small, lzma", t2-t1, t3-t2)

	# t1 = time.time()
	# ones_small_compressed = mt_gzip_compressor.compress([ones_small])
	# print("compress")
	# t2 = time.time()
	# _ = mt_gzip_compressor.decompress(ones_small_compressed)
	# print("decompress")
	# t3 = time.time()
	# results[3,3,0] = t2-t1
	# results[3,3,1] = t3-t2 
	# print("ones small, mt_gzip", t2-t1, t3-t2)

	# ## ones med
	# t1 = time.time()
	# ones_med_compressed = pickle_compressor.compress([ones_med])
	# t2 = time.time()
	# _ = pickle_compressor.decompress(ones_med_compressed)
	# t3 = time.time()
	# results[0,4,0] = t2-t1
	# results[0,4,1] = t3-t2 
	# print("ones med, pickle", t2-t1, t3-t2)

	# t1 = time.time()
	# ones_med_compressed = bz2_compressor.compress([ones_med])
	# t2 = time.time()
	# _ = bz2_compressor.decompress(ones_med_compressed)
	# t3 = time.time()
	# results[1,4,0] = t2-t1
	# results[1,4,1] = t3-t2 
	# print("ones med, bz2", t2-t1, t3-t2)

	# t1 = time.time()
	# ones_med_compressed = lzma_compressor.compress([ones_med])
	# t2 = time.time()
	# _ = lzma_compressor.decompress(ones_med_compressed)
	# t3 = time.time()
	# results[2,4,0] = t2-t1
	# results[2,4,1] = t3-t2 
	# print("ones med, lzma", t2-t1, t3-t2)

	# t1 = time.time()
	# ones_med_compressed = mt_gzip_compressor.compress([ones_med])
	# print("compress")
	# t2 = time.time()
	# _ = mt_gzip_compressor.decompress(ones_med_compressed)
	# print("decompress")
	# t3 = time.time()
	# results[3,4,0] = t2-t1
	# results[3,4,1] = t3-t2 
	# print("ones med, mt_gzip", t2-t1, t3-t2)

	## ones large
	t1 = time.time()
	ones_large_compressed = pickle_compressor.compress([ones_large])
	t2 = time.time()
	_ = pickle_compressor.decompress(ones_large_compressed)
	t3 = time.time()
	results[0,5,0] = t2-t1
	results[0,5,1] = t3-t2 
	print("ones large, pickle", t2-t1, t3-t2)

	t1 = time.time()
	ones_large_compressed = bz2_compressor.compress([ones_large])
	t2 = time.time()
	_ = bz2_compressor.decompress(ones_large_compressed)
	t3 = time.time()
	results[1,5,0] = t2-t1
	results[1,5,1] = t3-t2  
	print("ones large, bz2", t2-t1, t3-t2)

	t1 = time.time()
	ones_large_compressed = lzma_compressor.compress([ones_large])
	t2 = time.time()
	_ = lzma_compressor.decompress(ones_large_compressed)
	t3 = time.time()
	results[2,5,0] = t2-t1
	results[2,5,1] = t3-t2 
	print("ones large, lzma", t2-t1, t3-t2)

	t1 = time.time()
	ones_large_compressed = mt_gzip_compressor.compress([ones_large])
	print("compress")
	t2 = time.time()
	_ = mt_gzip_compressor.decompress(ones_large_compressed)
	print("decompress")
	t3 = time.time()
	results[3,5,0] = t2-t1
	results[3,5,1] = t3-t2 
	print("ones large, mt_gzip", t2-t1, t3-t2)

	# ## arange small
	# t1 = time.time()
	# arange_small_compressed = pickle_compressor.compress([arange_small])
	# t2 = time.time()
	# _ = pickle_compressor.decompress(arange_small_compressed)
	# t3 = time.time()
	# results[0,6,0] = t2-t1
	# results[0,6,1] = t3-t2 
	# print("arange small, pickle", t2-t1, t3-t2)

	# t1 = time.time()
	# arange_small_compressed = bz2_compressor.compress([arange_small])
	# t2 = time.time()
	# _ = bz2_compressor.decompress(arange_small_compressed)
	# t3 = time.time()
	# results[1,6,0] = t2-t1
	# results[1,6,1] = t3-t2 
	# print("arange small, bz2", t2-t1, t3-t2)

	# t1 = time.time()
	# arange_small_compressed = lzma_compressor.compress([arange_small])
	# t2 = time.time()
	# _ = lzma_compressor.decompress(arange_small_compressed)
	# t3 = time.time()
	# results[2,6,0] = t2-t1
	# results[2,6,1] = t3-t2 
	# print("arange small, lzma", t2-t1, t3-t2)

	# t1 = time.time()
	# arange_small_compressed = mt_gzip_compressor.compress([arange_small])
	# print("compress")
	# t2 = time.time()
	# _ = mt_gzip_compressor.decompress(arange_small_compressed)
	# print("decompress")
	# t3 = time.time()
	# results[3,6,0] = t2-t1
	# results[3,6,1] = t3-t2 
	# print("arange small, mt_gzip", t2-t1, t3-t2)

	# ## arange med
	# t1 = time.time()
	# arange_med_compressed = pickle_compressor.compress([arange_med])
	# t2 = time.time()
	# _ = pickle_compressor.decompress(arange_med_compressed)
	# t3 = time.time()
	# results[0,7,0] = t2-t1
	# results[0,7,1] = t3-t2 
	# print("arange med, pickle", t2-t1, t3-t2)

	# t1 = time.time()
	# arange_med_compressed = bz2_compressor.compress([arange_med])
	# t2 = time.time()
	# _ = bz2_compressor.decompress(arange_med_compressed)
	# t3 = time.time()
	# results[1,7,0] = t2-t1
	# results[1,7,1] = t3-t2 
	# print("arange med, bz2", t2-t1, t3-t2)

	# t1 = time.time()
	# arange_med_compressed = lzma_compressor.compress([arange_med])
	# t2 = time.time()
	# _ = lzma_compressor.decompress(arange_med_compressed)
	# t3 = time.time()
	# results[2,7,0] = t2-t1
	# results[2,7,1] = t3-t2 
	# print("arange med, lzma", t2-t1, t3-t2)

	# t1 = time.time()
	# arange_med_compressed = mt_gzip_compressor.compress([arange_med])
	# print("compress")
	# t2 = time.time()
	# _ = mt_gzip_compressor.decompress(arange_med_compressed)
	# print("decompress")
	# t3 = time.time()
	# results[3,7,0] = t2-t1
	# results[3,7,1] = t3-t2 
	# print("arange med, mt_gzip", t2-t1, t3-t2)

	## arange large
	t1 = time.time()
	arange_large_compressed = pickle_compressor.compress([arange_large])
	t2 = time.time()
	_ = pickle_compressor.decompress(arange_large_compressed)
	t3 = time.time()
	results[0,8,0] = t2-t1
	results[0,8,1] = t3-t2 
	print("arange large, pickle", t2-t1, t3-t2)

	t1 = time.time()
	arange_large_compressed = bz2_compressor.compress([arange_large])
	t2 = time.time()
	_ = bz2_compressor.decompress(arange_large_compressed)
	t3 = time.time()
	results[1,8,0] = t2-t1
	results[1,8,1] = t3-t2 
	print("arange large, bz2", t2-t1, t3-t2)

	t1 = time.time()
	arange_large_compressed = lzma_compressor.compress([arange_large])
	t2 = time.time()
	_ = lzma_compressor.decompress(arange_large_compressed)
	t3 = time.time()
	results[2,8,0] = t2-t1
	results[2,8,1] = t3-t2 
	print("arange large, lzma", t2-t1, t3-t2)

	t1 = time.time()
	arange_large_compressed = mt_gzip_compressor.compress([arange_large])
	print("compress")
	t2 = time.time()
	_ = mt_gzip_compressor.decompress(arange_large_compressed)
	print("decompress")
	t3 = time.time()
	results[3,8,0] = t2-t1
	results[3,8,1] = t3-t2 
	print("arange large, mt_gzip", t2-t1, t3-t2)

	# np.save("results/compression_test_kub2.npy", results)
