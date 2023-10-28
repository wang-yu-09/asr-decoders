# Deployable End-to-End ASR Decoders

All decoders are written in C++. besides using C++ interface directly, we also use pybind11 to provide python interface, which can be called in python directly after compilation.

These are currently provided decoders and related tools and their usage can be found in the demos under `unit_test`.

| Decoder    | Library     | Description     |
| -------- | -------- | -------- |
| `LexiconDecoder` | flashlight | This is the `LexiconDecoder` extracted from `flashlight`. It first builds a lexicon trie using a pronunciation lexicon, and then synthesizes words during a beam search while scoring them using a word-level language model. |
| `GreedyDecoder` | rmai | Greedy search algorithm.  |
| `PrefixBeamSearchDecoder` | rmai | Improved prefix beam search algorithm. |
| `LexiconTransducer` | rmai | A decoder for converting sub-word sequences into word sequences. |
| `FasterDecoder` | eesen, kaldi | This is a WFST decoder based on `EESEN`, using TLG decoding graph. This algorithm requires `EESEN` and `OPENBLAS` to be installed and compiled first. The method of installing, compiling and constructing the TLG decoding graph is written later. |

Other decoders are still being developed, and will be added to the library in the future.

--------------------------------------------
## How to compile and use

### 1. Configure some paths in `build.sh`
If compiling `flashlight` or `rmai`, you need to specify the path to Kenlm:
```bash
Kenlm_dir=/path/to/Kenlm
```
If compiling `eesen`, you need to specify `Essen` and `OpenBLAS`.
```bash
Essen_dir=/path/to/Eesen
Openblas_dir=/path/to/Kaldi/tools/OpenBLAS/install

```

### 2. Compile in the current path:
```bash
bash /path/to/build.sh
```
A library file named `*.so` is generated after the compilation.

### 3. Import libraries in python:
```python
from rmai_decoder import GreedyDecoder 
```
Please refer to the demos in `unit_test` for specific usage.

---------------------------------------------
## Folder

| Folder    |  Description     |
| -------- | -------- | 
| `src` | C++ source code for all decoders. |
| `unit_test` | Source code for some unit tests.  |
| `unit_test_data` | Data used in unit testing. |
| `example` | Some large test cases. |
| `tools` | Some other tools. |

---------------------------------------------
## Installation of Eesen and OpenBLAS

### 1. Clone EESEN from github
```bash
glone https://github.com/srvk/eesen.git Eesen
```
### 2. Compile the dependencies and install openblas
There is an `install_openblas.sh` file in the `tools` folder. Copy this script to `Eesen/tools` and start compiling and installing.
```bash
cd Eesen/tools
make
bash install_openblas.sh
```
### 3. Compile Eesen
We only use a decoder, so we don't need a CUDA. Also, you have to specify the path to OpenBLAS when compiling.
```bash
cd Eesen/src
./configure --shared --use-cuda=no --openblas-root=/path/to/OpenBLAS/install
make
```

---------------------------------------------
## build TLG.fst (Token-Lexicon-Grammar) graph

### 1. Create workshop
Switch to `Eesen/asr_egs` and create a project folder, e.g. `demo`, and then create two soft links.
```bash
cd Eesen/asr_egs
mkdir demo && cd demo
ln -s /path/to/Eesen/asr_egs/librispeech/utils utils
ln -s /path/to/Eesen/asr_egs/librispeech/steps steps
```

### 2. Preparation
Create a `src` path under `demo`, and put the following three files under the path:  
1) `lexicon.txt`: Word->Subword Dictionary, e.g. pronunciation dictionary. The format of each line is e.g.: `打开 da kai`  
2) `units.txt`: The id of blank must be 0, and this lexicon contains all subword units except blank. When the lexicon is subsequently created, a `<eps>` and `<blk>` (the blank symbol) are inserted at the top, so the resulting lexicon actually has one more category than the output of the CTC, and the IDs are staggered backwards by one, and the id of blank becomes a 1. But don't worry, where I take the probabilities in the decoder, I subtracted a 1 from the IDs to cancel out this error. The format of each line is as follows: `da`  
3) `lexicon_numbers.txt`: Replacing the subword in the lexicon with an integer ID is sufficient. One of the bash scripts used below checks for the existence of this file, but we won't actually be using it throughout the process, so creating an empty file will do the trick.  

In addition, you need to prepare an arpa language model:
1) `n-gram.arpa`: Word-level ngram model, arpa format.

### 3. Further processing of the lexicon and production of L.fst
After the above 4 files are prepared, the following script is used to further process the dictionary and generate `L.fst`.
```bash
utils/ctc_compile_dict_token.sh --dict-type "phn" src lang/tmp lang
```
For English, you need to specify `--dict-type "char" --space-char "|"` if you are using an alphabetic dictionary. 
The final generated file is in the `lang` directory. 
`lang/tokens.txt` is the actual subword dictionaries  and their IDs used in the final decoded map. 
`lang/words.txt` represents the actual word dictionaries used.
Note: The word dictionary passed in during decoding should be `lang/words.txt`.

Essen uses python2, which can be specified by displaying `utils/ctc_compile_dict_token.sh` where the python script is called.

```bash
python2 utils/ctc_token_fst.py ...
```

### 4. Synthesize the final TLG decoding map
Under `tools`, there is a reference script for synthesizing TLG.fst, which you can copy, specify the root directory of `openfst`, and you're good to go!
```bash
cp tools/compile_tlg_graph.sh /path/to/Eesen/asr_egs/demo
cd /path/to/Eesen/asr_egs/demo
bash compile_tlg_graph.sh lang /path/to/n-gram.arpa graph
```
The final generated decoded graph is located at `graph/TLG.fst`.

### 5. Calculate prior probabilities (not required)
Using the training dataset to generate integer labels, the format of the generated labels is as follows: 
```text
TV1_000160001_0000 61 37 206 194 61 176 230 142
```
The first column is the utterace ID, followed by the result of converting the corresponding text label to CTC's class ID. 
Note that the sym2int lexicon used in the conversion should be the original CTC lexicon, which is the one with blank id=0, not the `tokens.txt` generated during the composition process.

After preparing this file use the following command to generate the statistics file:
```bash
cat $label | awk '{line=$0; gsub(" "," 0 ",line); print line " 0";}' | Eesen2/src/decoderbin/analyze-counts --verbose=1 --binary=false ark:- graph/label.counts >& graph/log/compute_label_counts.log
```
The final statistics file is located in `graph/label.counts`. An example of how to use it is given below:
```python
import numpy as np

def compute_class_prior(class_count_file:str, prior_cutoff:float=1e-10, blank_scale:float=1.0) -> np.ndarray:

    with open(class_count_file, "r", encoding="utf-8") as fr:
        line = fr.readline().strip().strip("[]").strip()
        line = line.split()
    
    counts = np.array(line, dtype="float32")
    #mask = np.zeros_like(counts)

    #mask[ counts<prior_cutoff ] = np.float.min
    counts[counts<prior_cutoff] = prior_cutoff

    if blank_scale != 1.0:
        counts[0] *= blank_scale
    
    return np.log(counts/np.sum(counts))

def apply_prior(prob:np.ndarray, prior:np.ndarray, prior_scale:float=1.0) -> np.ndarray:
    return prob - prior_scale * prior

prior_bias = compute_class_prior("graph/label.counts")

# Assume that log_prob is a two-dimensional matrix that has been processed by log_softmax.
new_log_prob = apply_prior(log_prob, prior_bias)
```



