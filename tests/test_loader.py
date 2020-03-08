import time
import hanlperceptron

model_path = '../data/'

segmenter1 = hanlperceptron.Segmenter()
segmenter2 = hanlperceptron.Segmenter()

# origin model file
start = time.time()
segmenter1.load(model_path+'large/cws.bin')
print('Spend: %f(s)' % (time.time() - start))

# save model file
segmenter1.save('cws.pkl')

# load new model file
start = time.time()
segmenter2.load('cws.pkl')
print('Spend: %f(s)' % (time.time() - start))
