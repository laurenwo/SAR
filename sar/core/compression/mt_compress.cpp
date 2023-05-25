#include <torch/extension.h>
// #include <pybind11/stl.h>
// #include <pybind11/stl_bind.h>
#include <iostream>
#include <string>
#include <vector>
#include <omp.h>

#include "gzip_compress.hpp"
#include "gzip_decompress.hpp"

// PYBIND11_MAKE_OPAQUE(std::vector<py::bytes>);

std::vector<py::bytes> mt_compress(
    std::vector<py::bytes> partitioned_tensors
){
    std::vector<py::bytes> compressed_tensors_l(partitioned_tensors.size());
#pragma omp parallel for
    for(int i = 0; i < partitioned_tensors.size(); i++){
        std::string data = partitioned_tensors[i];
        const char * pointer = data.data();
        std::size_t size = data.size();
        compressed_tensors_l[i] = gzip::compress(pointer, size);
    }
    return compressed_tensors_l;
}

std::vector<py::bytes> mt_decompress(
    std::vector<py::bytes> channel_feat
){
    std::vector<py::bytes> tensors_l(channel_feat.size());
#pragma omp parallel for
    for(int i = 0; i < channel_feat.size(); i++){
        std::string data = channel_feat[i];
        const char * pointer = data.data();
        std::size_t size = data.size();
         tensors_l[i] = gzip::decompress(pointer, size);
    }
    return tensors_l;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mt_compress", &mt_compress, "Multi-threaded compression");
  m.def("mt_decompress", &mt_decompress, "Multi-threaded decompression");
//   py::bind_vector<std::vector<py::bytes>>(m, "byte_vector");
}