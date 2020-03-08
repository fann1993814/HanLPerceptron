import hanlperceptron

model_path = '../data/'

segmenter = hanlperceptron.Segmenter(model_path+'pku199801/cws.bin')

text = '芋頭牛奶霜淇淋使用大甲的新鮮芋頭及澳洲香濃牛奶手工製作'

res_viterbi = segmenter.segment(text, viterbi=True)
res_greedy = segmenter.segment(text, viterbi=False)

print(res_viterbi)
print(res_greedy)
