import hanlperceptron

# Load model

model_path = '../data/'
model_path = '/Users/Jason/Downloads/data-for-1.7.5/data/model/perceptron/'
segmenter = hanlperceptron.Segmenter()
postager = hanlperceptron.POSTagger()

segmenter.load(model_path+'pku199801/cws.bin')
postager.load(model_path+'pku199801/pos.bin')

# Oringinal model

text = '畢卡索是堂何塞路伊思布拉斯可和瑪莉亞畢卡索洛佩茲的第一個孩子。'

seg_res = segmenter.segment(text)
pos_res = postager.tagging(seg_res)
print(seg_res)
print(pos_res)

# Custom Dictionary
# 堂何塞路伊思布拉斯可 1 nr
# 瑪莉亞畢卡索洛佩茲 1 nr
segmenter.load_custom_dict('dict.txt')

seg_res = segmenter.segment(text)
pos_res = postager.tagging(seg_res)
print(seg_res)
print(pos_res)
