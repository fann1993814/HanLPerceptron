import hanlperceptron

model_path = '../data/'

segmenter = hanlperceptron.Segmenter(model_path+'large/cws.bin')
postager = hanlperceptron.POSTagger(model_path+'pku1998/pos.bin')
nerecognizer = hanlperceptron.NERecognizer(model_path+'pku1998/ner.bin')

text = '大西洋和太平洋'

seg_res = segmenter.segment(text)
pos_res = postager.tag(seg_res)
ner_res = nerecognizer.recognize(seg_res, pos_res)

print(seg_res)
print(pos_res)
print(ner_res)
