# 语音识别解码器

所有解码器底层都是C++写的, 除了直接使用C++接口外, 我们也使用pybind11提供了python的接口, 只要编译后就可以直接在python中调用。

目前提供的解码器和相关工具，他们的使用方法可以参考`unit_test`下的demo:

| 解码器    | 库名     | 备注     |
| -------- | -------- | -------- |
| `LexiconDecoder` | flashlight | 抽取的flashlight的`LexiconDecoder`, 先使用发音词典构建词典树, 在beam search过程中一边合成单词, 一边使用单词级别的语言模型打分. |
| `GreedyDecoder` | rmai | 贪婪解码.  |
| `PrefixBeamSearchDecoder` | rmai | 前缀束解码. |
| `LexiconTransducer` | rmai | 将子词序列转换成单词序列的解码器. |
| `FasterDecoder` | eesen, kaldi | 基于EESEN的WFST解码器, 使用TLG解码图. 这个算法需要先安装和编译EESEN和OPENBLAS, 安装编译和构建TLG解码图方法写在后面. |

其他的解码器还在开发和实验中, 后期会逐渐加入到这个库里面。

--------------------------------------------
## 编译和使用

### 1. 设置`build.sh`里面路径
如果编译`flashlight`或者`rmai`需要指定Kenlm:
```bash
Kenlm_dir=/path/to/Kenlm
```
如果编译`eesen`,需要指定Essen和OpenBLAS:
```bash
Essen_dir=/path/to/Eesen
Openblas_dir=/path/to/Kaldi/tools/OpenBLAS/install

```

### 2. 在当前路径编译
编译结束后会生成一个名为`*.so`的库文件.
```bash
bash /path/to/build.sh
```

### 3. 在python中import就可以了
具体用法参考单元测试的demo.
```python
from rmai_decoder import GreedyDecoder 
```

---------------------------------------------
## 目录说明

1. `src`: 我们自己的一些解码器的C++源码
2. `unit_test`: 一些单元测试的C++或python源码
3. `unit_test_data`: 单元测试时使用到的数据
4. `example`: 一些大型测试案例
5. `tools`: 一些其他工具

---------------------------------------------
## 安装Eesen和OpenBLAS

### 1. 克隆EESEN库
```bash
glone https://github.com/srvk/eesen.git Eesen
```
### 2. 编译依赖工具, 同时安装openblas
在我提供`tools`文件夹下有一个`install_openblas.sh`的脚本，拷贝到`Eesen/tools`下, 然后开始编译和安装
```bash
cd Eesen/tools
make
bash install_openblas.sh
```
### 3. 编译Eesen库
我们只使用decoder所以无需cuda, 并指定OpenBLAS路径
```bash
cd Eesen/src
./configure --shared --use-cuda=no --openblas-root=/path/to/OpenBLAS/install
make
```

---------------------------------------------
## 合成 TLG.fst (Token-Lexicon-Grammar) 图

### 1. 创建工作区
切换到`Eesen/asr_egs`下创建一个项目文件夹, 例如`demo`, 然后创建两个软链接
```bash
cd Eesen/asr_egs
mkdir demo && cd demo
ln -s /path/to/Eesen/asr_egs/librispeech/utils utils
ln -s /path/to/Eesen/asr_egs/librispeech/steps steps
```

### 2. 准备资源
在`demo`下创建一个`src`路径, 并在路径下放入下面3个文件:  
1) `lexicon.txt`: 单词->子词 词典, 例如发音词典, 每行格式例如: `打开 da kai`  
2) `units.txt`: blank的id必须是0, 这个词典包含了除blank的所有子词单元, 每行为一个子词单元。在后续创建词典时,会在最前端插入一个`<eps>`和`<blk>`(blank符号), 因此最后生成的词典实际上比CTC的输出类别多一个, 并且ID向后错开了一个, blank的id变成了1, 但是不用担心, 在解码器中取概率的地方, 我在ID上减了一个1来抵消这个误差。每行格式例如: `da`  
3) `lexicon_numbers.txt`: 将lexicon中的子词替换成整型ID就可以了, 在下面用到的一个bash脚本会检查这个文件是否存在, 但我们整个过程中实际上并不会使用它, 因此创建一个空文件就可以了.  

另外你需要准备一个arpa语言模型:
1) `n-gram.arpa`: 单词级别ngram模型, arpa格式

### 3. 进一步处理词典并制作L.fst
上面的4个文件准备完成后, 使用下面的脚本进一步处理词典并生成`L.fst` 
```bash
utils/ctc_compile_dict_token.sh --dict-type "phn" src lang/tmp lang
```
对于英文如果使用的字母词典, 指定`--dict-type "char" --space-char "|"`, 最终生成的文件在`lang`目录下。`lang/tokens.txt`是最后的解码图实际使用的子词词典和他们的ID, `lang/words.txt`代表实际使用的单词词典.
注意: 在解码时传入的单词词典应该使用`lang/words.txt`.
Essen使用python2, 可以把`utils/ctc_compile_dict_token.sh`中调用python脚本的地方显示地指定为python2
```bash
python2 utils/ctc_token_fst.py ...
```

### 4. 合成最终的TLG解码图
在`tools`下是一个合成TLG.fst的参考脚本,你可以复制过去, 指定`openfst`的根目录，然后就可以用了
```bash
cp tools/compile_tlg_graph.sh /path/to/Eesen/asr_egs/demo
cd /path/to/Eesen/asr_egs/demo
bash compile_tlg_graph.sh lang /path/to/n-gram.arpa graph
```
最终生成的解码图位于`graph/TLG.fst`.

### 5. 统计先验概率（非必须）
使用训练数据集生成整型标签, 生成的标签格式如下
```text
TV1_000160001_0000 61 37 206 194 61 176 230 142
```
第一列为 utterace ID, 后面的是将对应的文本标签转换为CTC的class ID的结果, 注意, 转换时使用的 sym2int 词典应该是CTC的原始的词典, 也就是说blank id=0的那个词典, 而不是在构图过程中生成的`tokens.txt`.

准备好这个文件后使用下面的命令生成统计文件:
```bash
cat $label | awk '{line=$0; gsub(" "," 0 ",line); print line " 0";}' | Eesen2/src/decoderbin/analyze-counts --verbose=1 --binary=false ark:- graph/label.counts >& graph/log/compute_label_counts.log
```
最后生成的统计文件位于`graph/label.counts`. 下面给出一个使用方法例子:
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

# 假设 log_prob 是经历了log softmax的二维矩阵
new_log_prob = apply_prior(log_prob, prior_bias)

# 接下来就可以用这个概率去解码了
```



