import kenlm

src_fname = "/path/to/word_to_phoneme.txt"
kenlm_fname = "/path/to/word-2g.bin"
out_fname = "lexiconp.txt"

model = kenlm.Model(kenlm_fname)

new_lines = []
score = model.score("<unk>", bos=False, eos=False)
new_lines.append( f"<unk> {score} <unk>" )
with open(src_fname,"r") as fr:
    for line in fr:
        line = line.strip().split(maxsplit=1)
        # print(line)
        word = line[0]
        phones = line[1]
        score = model.score(word, bos=False, eos=False)
        new_lines.append( f"{word} {score} {phones}" )

with open(out_fname,"w") as fw:
    fw.write( "\n".join(new_lines) )

