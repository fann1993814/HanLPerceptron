import hanlperceptron

# Load model

model_path = '../data/'

segmenter = hanlperceptron.Segmenter()
postager = hanlperceptron.POSTagger()

segmenter.load(model_path+'pku199801/cws.bin')
postager.load(model_path+'pku199801/pos.bin')

# Oringinal model

text = '畢卡索是堂何塞路伊思布拉斯可和瑪莉亞畢卡索洛佩茲的第一個孩子。'

seg_res = segmenter._segment(text)
pos_res = postager._tagging(seg_res)
print(seg_res)
print(pos_res)

# Customise model
# 堂何塞路伊思布拉斯可 nr
# 瑪莉亞畢卡索洛佩茲 nr
segmenter.add_word('堂何塞路伊思布拉斯可')
segmenter.add_word('瑪莉亞畢卡索洛佩茲')
postager.add_tag('堂何塞路伊思布拉斯可', 'nr')
postager.add_tag('瑪莉亞畢卡索洛佩茲', 'nr')
seg_res = segmenter.segment(text)
pos_res = postager.tagging(seg_res)
print(seg_res)
print(pos_res)
