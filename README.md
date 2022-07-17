# DPAN: look back Again Dual Parallel Attention Network for Accurate and Robust Scene Text Recognition
Unofficial Implementation of paper "look back Again: Dual Parallel Attention Network for Accurate and Robust Scene Text Recognition"

## Training:
Add training and validation images to `dataset/train_data`.

Generate `train.txt` and `val.txt` labels in the following format:

Image filename followed by tab seperator followed by ground truth text label.

Example:
```train.txt
img1.png	synthesizer
img2.png	waterpark
img3.png	pokemon
```
(TAB seperator)

To train, use the `train.py`:
```
python train.py --root dataset/train_data --train_csv train.txt --val_csv train.txt
```

## Acknowledgements:
### Paper:
```
@inproceedings{10.1145/3460426.3463674,
author = {Fu, Zilong and Xie, Hongtao and Jin, Guoqing and Guo, Junbo},
title = {Look Back Again: Dual Parallel Attention Network for Accurate and Robust Scene Text Recognition},
year = {2021},
isbn = {9781450384636},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3460426.3463674},
doi = {10.1145/3460426.3463674},
abstract = {Nowadays, it is a trend that using a parallel-decoupled encoder-decoder (PDED) framework in scene text recognition for its flexibility and efficiency. However, due to the inconsistent information content between queries and keys in the parallel positional attention module (PPAM) used in this kind of framework(queries: position information, keys: context and position information), visual misalignment tends to appear when confronting hard samples(e.g., blurred texts, irregular texts, or low-quality images). To tackle this issue, in this paper, we propose a dual parallel attention network (DPAN), in which a newly designed parallel context attention module (PCAM) is cascaded with the original PPAM, using linguistic contextual information to compensate for the information inconsistency between queries and keys. Specifically, in PCAM, we take the visual features from PPAM as inputs and present a bidirectional language model to enhance them with linguistic contexts to produce queries. In this way, we make the information content of the queries and keys consistent in PCAM, which helps to generate more precise visual glimpses to improve the entire PDED framework's accuracy and robustness. Experimental results verify the effectiveness of the proposed PCAM, showing the necessity of keeping the information consistency between queries and keys in the attention mechanism. On six benchmarks, including regular text and irregular text, the performance of DPAN surpasses the existing leading methods by large margins, achieving new state-of-the-art performance. The code is available on urlhttps://github.com/Jackandrome/DPAN.},
booktitle = {Proceedings of the 2021 International Conference on Multimedia Retrieval},
pages = {638–644},
numpages = {7},
keywords = {text recognitoin, attention mechanism, languge model},
location = {Taipei, Taiwan},
series = {ICMR '21}
}
```

The github was built by modifying https://github.com/chenjun2hao/Bert_OCR.pytorch
