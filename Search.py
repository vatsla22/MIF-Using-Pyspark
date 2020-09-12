from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.functions import regexp_replace,trim,col,lower,split,size,count,countDistinct,log10,explode,round
from pyspark.ml.feature import Tokenizer
from pyspark.sql.functions import sum
from pyspark.sql.functions import desc

def removepunctuations(col):
    return trim(lower(regexp_replace(col,'[^\sa-zA-Z0-9]',''))).alias('textentry')
    
def sentence_data(df_data):
    df_data2 = df_data.select(df_data._id , removepunctuations(df_data.text_entry))
    only_words = Tokenizer(inputCol = 'textentry' , outputCol="words")
    df_data3 = only_words.transform(df_data2)
    return df_data3
    
def indexing(df_data):
    df_data2 = df_data.select(df_data._id , removepunctuations(df_data.text_entry))
    only_words = Tokenizer(inputCol = 'textentry' , outputCol="words")
    df_data3 = only_words.transform(df_data2)
    df_data4 = df_data3.select(df_data3._id,df_data3.textentry,explode(df_data3.words).alias('token_words'))
    term_freq = df_data4.groupBy("_id","token_words").agg(count("token_words").alias("TF"))
    doc_freq = df_data4.groupBy("token_words").agg(countDistinct("_id").alias("DF"))
    idf_calc = doc_freq.withColumn('idf' , (111396.0)/doc_freq['DF'])
    idf_calc = idf_calc.withColumn("IDF" , log10("idf"))
    tf_idf = term_freq.join(idf_calc,"token_words","left").withColumn("TF-IDF",col("TF")*col("IDF"))
    return tf_idf

def search_words(query,N,tfidf,df_data):
    q= query.split(' ')
    n= len(q)
    df_data3 = sentence_data(df_data)
    data = tfidf.filter(col("token_words").isin([x for x in q]))
    total_tfidf = data.groupBy("_id").agg(sum("TF-IDF").alias("total_sum"))
    occurance = data.groupBy("_id").count().toDF('_id','occurance')
    sum_data = total_tfidf.join(occurance,"_id","inner")
    sum_data = sum_data.withColumn("score",col("occurance")/n * col("total_sum"))
    sum_data = sum_data.join(df_data3,"_id","inner")
    sum_data = sum_data.select(sum_data._id,round(sum_data.score,3).alias("score"),sum_data.textentry)
    sum_data = sum_data.sort(desc("score"))
    answer = sum_data.limit(N).rdd.map(lambda x: (x._id, x.score, str(x.textentry))).collect()
    for tuple in answer:
        print(tuple)

def main(sc):
    context = SQLContext(sc)
    df_data = context.read.json("/user/maria_dev/vatsla/finalterm/shakespeare_full.json")
    tfidfdata = indexing(df_data)
    result = search_words("to be or not",1,tfidfdata,df_data)
    #result = search_words("to be or not",3,tfidfdata,df_data)
    #result = search_words("to be or not",5,tfidfdata,df_data)
    #result = search_words("so far so",1,tfidfdata,df_data)
    #result = search_words("so far so",3,tfidfdata,df_data)
    #result = search_words("so far so",5,tfidfdata,df_data)
    #result = search_words("if you said so",1,tfidfdata,df_data)
    #result = search_words("if you said so",3,tfidfdata,df_data)
    #result = search_words("if you said so",5,tfidfdata,df_data)
    
if __name__  == "__main__":
    conf = SparkConf().setAppName("MyApp")
    sc = SparkContext(conf = conf)
    main(sc)
    sc.stop()

#spark-submit --master yarn-client --executor-memory 512m --num-executors 3 --executor-cores 1 --driver-memory 512m search.py