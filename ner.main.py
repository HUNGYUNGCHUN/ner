import pandas as pd
import torch
from src import inference_api
import src.config as config

# 讀取 CSV 檔案
df = pd.read_csv('top250(tfidf.semantic).csv', encoding='utf-8-sig')

# 顯示前 10 行
print(df.head(10))

# 設定設備
config.string_device = 'cuda' if torch.cuda.is_available() else 'cpu'
config.device = torch.device(config.string_device)

# 載入模型
model, tokenizer = inference_api.load_model("clw8998/Product-Name-NER-model", device=config.device)

# Relevancy 計算函數
def ner_relevancy(df, index, all_results, check_att, margin):
   try:
       query_key = df.loc[index, '搜尋詞'].lower().strip()
       tfidf_key = df.loc[index, 'tf-idf'].lower().strip()
       semantic_key = df.loc[index, 'semantic_model'].lower().strip()

       # 確認所有鍵在 all_results 中
       query_tags_dict = all_results.get(query_key, {})
       tfidf_tags_dict = all_results.get(tfidf_key, {})
       semantic_tags_dict = all_results.get(semantic_key, {})

       # 當所有標籤字典為空時，忽略該行
       if not query_tags_dict and not tfidf_tags_dict and not semantic_tags_dict:
           return df  # 如果所有標籤都無法辨識，則忽略此行

       # 提取標籤
       query_tags_pool = set(ent[0] for ents in query_tags_dict.values() for ent in ents if ent)
       tfidf_tags_pool = set(ent[0] for ents in tfidf_tags_dict.values() for ent in ents if ent)
       semantic_tags_pool = set(ent[0] for ents in semantic_tags_dict.values() for ent in ents if ent)

       # 計算 query_tags_pool 與 tfidf_tags_pool 的交集大小
       print(f"Index: {index}, 交集大小 (query & tfidf): {len(query_tags_pool & tfidf_tags_pool)}")
       print(f"Index: {index}, 交集大小 (query & tfidf): {len(query_tags_pool & semantic_tags_pool)}")



       # 輸出標籤
       print(f"Index: {index}")
       print(f"搜尋詞: {query_key}, 標籤: {query_tags_pool}")
       print(f"TF-IDF: {tfidf_key}, 標籤: {tfidf_tags_pool}")
       print(f"Semantic: {semantic_key}, 標籤: {semantic_tags_pool}")
       print("-" * 50)

       # 計算相關性
       if len(query_tags_pool & tfidf_tags_pool) >= len(check_att) * margin:
           df.loc[index, 'ner_relevancy_1'] = 2
       elif len(query_tags_pool & tfidf_tags_pool) >= len(check_att) * margin * 0.5:
           df.loc[index, 'ner_relevancy_1'] = 1
       else:
           df.loc[index, 'ner_relevancy_1'] = 0

       if len(query_tags_pool & semantic_tags_pool) >= len(check_att) * margin:
           df.loc[index, 'ner_relevancy_2'] = 2
       elif len(query_tags_pool & semantic_tags_pool) >= len(check_att) * margin * 0.5:
           df.loc[index, 'ner_relevancy_2'] = 1
       else:
           df.loc[index, 'ner_relevancy_2'] = 0

   except Exception as e:
       print(f"Error processing index {index}: {e}")

   return df

check_att = ['品牌', '產品', '顏色', '適用物體、事件與場所', '功能與規格']
all_results = inference_api.get_ner_tags(
   model,
   tokenizer,
   list(set(df['搜尋詞'].tolist() + df['tf-idf'].tolist() + df['semantic_model'].tolist())),
   check_att)

# 遍歷 DataFrame 並計算相關性
for index, row in df.iterrows():
   df = ner_relevancy(df, index, all_results, check_att, 0.3)

# 將結果保存為 CSV
df.to_csv('NER-relevancy-results_250.csv', index=False, encoding='utf-8-sig')
