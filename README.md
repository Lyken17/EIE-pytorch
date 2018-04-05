# EIE-pytorch
PyTorch implementation for _EIE: Efficient Inference Engine on Compressed Deep Neural Network_  https://arxiv.org/abs/1602.01528


## Pruning
Following the paper, I adopt fine-grade pruning for the model compression.

## Quantization
The original paper uses K-means to quantizise. Beside K-means, I also tried with DBSCAN algorithm, which appears certain improvement on the accuracy. 

## Huffman Coding
The huffman coding is embedded in `HDF5` .
