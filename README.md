# HanLPerceptron:
HanLPerceptron主要是移植自 [HanLP v1](https://github.com/hankcs/HanLP/tree/1.x) 中包含Perceptron模組相關的核心功能，其中包含中文斷詞、詞性標註、命名實體識別等常用分析模組，而不包括Online Learning相關功能。此外，同樣支持簡體中文和繁體中文的分析，以及自定義辭典的整合。HanLPerceptron是原生Python，更方便搭配Python開發的應用相關結合。

## 安裝
```bash
pip install hanlperceptron
```
* 需要採用Python 3.6以上版本
* 亦或下載 https://github.com/fann1993814/HanLPerceptron ，解壓縮執行 `python setup.py install`

## 資料包準備
請下載：[data.zip](http://nlp.hankcs.com/download.php?file=data) (註：使用資料請遵守 [HanLP](https://github.com/hankcs/HanLP/) 的授權規範)

亦或至 https://github.com/hankcs/HanLP/releases 下載v1.7以上的資料包

下載後解壓到任意目錄，請留意資料包中`data/model/perceptron`的目錄位置。

    perceptron
    │
    ├── ctb
    │   └── pos.bin
    ├── large
    │   └── cws.bin
    ├── pku1998
    │   ├── cws.bin
    │   ├── pos.bin
    │   └── ner.bin
    └── pku199801
        ├── cws.bin
        ├── pos.bin
        └── ner.bin

## 基本功能演示
```python
>>> import hanlperceptron
>>> segmenter = hanlperceptron.Segmenter(model_path+'large/cws.bin')
>>> postager = hanlperceptron.POSTagger(model_path+'pku1998/pos.bin')
>>> nerecognizer = hanlperceptron.NERecognizer(model_path+'pku1998/ner.bin')
>>> segmenter.segment('大西洋和太平洋')
['大西洋', '和', '太平洋']
>>> postager.tagging(['大西洋', '和', '太平洋'])
['ns', 'c', 'ns']
>>> nerecognizer.recognize(['大西洋', '和', '太平洋'], ['ns', 'c', 'ns'])
['S', 'O', 'S']
```
- 說明
  * model_path為資料包的絕對路徑
  * POSTagger的模型有兩種不同的選擇 1. ctb 2. pku，分別輸出的類型也不同
  * NERecognizer輸入為斷詞結果和標註結果，其中標註結果須為pku的詞性準則
  ---
  * POSTagger的標簽集合如下表：

  | 標簽 | 含義     | 標簽 | 含義     | 標簽 | 含義     | 標簽 | 含義     |
  | ---- | -------- | ---- | -------- | ---- | -------- | ---- | -------- |
  | n    | 普通名詞 | f    | 方位名詞 | s    | 處所名詞 | t    | 時間     |
  | nr   | 人名     | ns   | 地名     | nt   | 機構名   | nw   | 作品名   |
  | nz   | 其他專名 | v    | 普通動詞 | vd   | 動副詞   | vn   | 名動詞   |
  | a    | 形容詞   | ad   | 副形詞   | an   | 名形詞   | d    | 副詞     |
  | m    | 數量詞   | q    | 量詞     | r    | 代詞     | p    | 介詞     |
  | c    | 連詞     | u    | 助詞     | xc   | 其他虛詞 | w    | 標點符號 |

## 解碼模式選擇
```python
>>> segmenter = hanlperceptron.Segmenter(model_path+'pku199801/cws.bin')
>>> segmenter.segment('芋頭牛奶霜淇淋使用大甲的新鮮芋頭及澳洲香濃牛奶手工製作', viterbi=True) #viterbi decode
['芋頭', '牛奶', '霜淇淋', '使用', '大甲', '的', '新鮮', '芋頭', '及', '澳洲', '香', '濃', '牛奶', '手工', '製作']
>>> segmenter.segment('芋頭牛奶霜淇淋使用大甲的新鮮芋頭及澳洲香濃牛奶手工製作', viterbi=False) #greedy decode
['芋頭', '牛', '奶', '霜', '淇', '淋', '使用', '大甲', '的', '新鮮', '芋頭', '及', '澳洲', '香', '濃', '牛奶', '手工', '製作']
```
- 說明
  * HanLP官方預設是採用viterbi解碼（精度高速度慢），然而考量速度因素，因此額外設計greedy解碼（精度略低速度快），讓用戶根據處理資料還有各自需求去調整。
  * 每個模組預設皆為viterbi解碼（Segmenter, POSTagger, NERecognizer）
  * Segmenter和POSTagger在預設狀態會使用CoreDictionary來先分析。
  
## 加速載入模型
```python
>>> import time
>>> segmenter1 = hanlperceptron.Segmenter()
>>> segmenter2 = hanlperceptron.Segmenter()
>>> start = time.time()
>>> segmenter1.load(model_path+'large/cws.bin')
>>> print('Spend: %f(s)' % (time.time() - start))
>>> Spend: 86.390805(s)
>>> segmenter1.save('cws.pkl') # save model file
>>> start = time.time()
>>> segmenter2.load('cws.pkl')
>>> print('Spend: %f(s)' % (time.time() - start))
>>> Spend: 2.371347(s)
```
- 說明
  * 由於原先是載入HanLP原始格式的資料包，需耗費大量時間進行剖析，經由重新儲存成HanLPerceptron的格式，載入時可減少剖析流程，大幅提升載入速度。（上方案例約為加速40倍）
  
## 自定義辭典演示
```python
>>> segmenter.segment('畢卡索是堂何塞路伊思布拉斯可和瑪莉亞畢卡索洛佩茲的第一個孩子。')
>>> ['畢卡索', '是', '堂何塞路', '伊', '思布拉斯可', '和', '瑪莉亞', '畢卡', '索洛佩茲', '的', '第一', '個', '孩子', '。']
>>> postager.tagging(['畢卡索', '是', '堂何塞路', '伊', '思布拉斯可', '和', '瑪莉亞', '畢卡', '索洛佩茲', '的', '第一', '個', '孩子', '。'])
>>> ['vn', 'v', 'q', 'j', 'v', 'c', 'ns', 'n', 'nr', 'u', 'm', 'q', 'n', 'w']
>>> # Custom Dictionary Format:
>>> # 堂何塞路伊思布拉斯可 1 nr
>>> # 瑪莉亞畢卡索洛佩茲 1 nr
>>> segmenter.load_custom_dict('dict.txt')
>>> segmenter.segment('畢卡索是堂何塞路伊思布拉斯可和瑪莉亞畢卡索洛佩茲的第一個孩子。')
>>> ['畢卡索', '是', '堂何塞路伊思布拉斯可', '和', '瑪莉亞畢卡索洛佩茲', '的', '第一', '個', '孩子', '。']
>>> postager.tagging(['畢卡索', '是', '堂何塞路伊思布拉斯可', '和', '瑪莉亞畢卡索洛佩茲', '的', '第一', '個', '孩子', '。'])
>>> ['vn', 'v', 'nr', 'c', 'nr', 'u', 'm', 'q', 'n', 'w']
```
- 說明
  * 載入自定義辭典可透過`load_custom_dict`完成針對Segmenter，同時也會載入詞性至POSTagger。
  * 辭典的格式參考[jieba-dict](https://github.com/fxsjy/jieba/raw/master/extra_dict/dict.txt.big)

## 加新詞或詞性
```python
>>> segmenter.segment('畢卡索是堂何塞路伊思布拉斯可和瑪莉亞畢卡索洛佩茲的第一個孩子。')
>>> ['畢卡索', '是', '堂何塞路', '伊', '思布拉斯可', '和', '瑪莉亞', '畢卡', '索洛佩茲', '的', '第一', '個', '孩子', '。']
>>> postager.tagging(['畢卡索', '是', '堂何塞路', '伊', '思布拉斯可', '和', '瑪莉亞', '畢卡', '索洛佩茲', '的', '第一', '個', '孩子', '。'])
>>> ['vn', 'v', 'q', 'j', 'v', 'c', 'ns', 'n', 'nr', 'u', 'm', 'q', 'n', 'w']
>>> # 堂何塞路伊思布拉斯可 1 nr
>>> # 瑪莉亞畢卡索洛佩茲 1 nr
>>> segmenter.add_word('堂何塞路伊思布拉斯可')
>>> segmenter.add_word('瑪莉亞畢卡索洛佩茲')
>>> ['畢卡索', '是', '堂何塞路伊思布拉斯可', '和', '瑪莉亞畢卡索洛佩茲', '的', '第一', '個', '孩子', '。']
>>> postager.add_tag('堂何塞路伊思布拉斯可', 'nr')
>>> postager.add_tag('瑪莉亞畢卡索洛佩茲', 'nr')
>>> postager.tagging(['畢卡索', '是', '堂何塞路伊思布拉斯可', '和', '瑪莉亞畢卡索洛佩茲', '的', '第一', '個', '孩子', '。'])
>>> ['vn', 'v', 'nr', 'c', 'nr', 'u', 'm', 'q', 'n', 'w']
```

## 授權

由於本項目主要參考至HanLP的諸多項目，因此延續採用與其同樣授權 **Apache License 2.0** 。然而資料包授權為HanLP，非為本項目相關，若要使用請遵守其要求規範。（请在产品说明中附加HanLP的链接和授权协议。HanLP受版权法保护，侵权必究。）

## 參考

感謝以下项目：

- [HanLP](https://github.com/hankcs/HanLP/)
- [jieba](https://github.com/fxsjy/jieba/)
