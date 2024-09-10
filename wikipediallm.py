import gradio as gr
import wikipediaapi
from sparkai.llm.llm import ChatSparkLLM, ChunkPrintHandler
from sparkai.core.messages import ChatMessage
from sparkai.errors import SparkAIConnectionError

# 设置讯飞星火API密钥和URL
SPARKAI_URL = 'wss://spark-api.xf-yun.com/v3.5/chat'
SPARKAI_APP_ID = 'YOUR_SPARKAI_APP_ID'
SPARKAI_API_SECRET = 'YOUR_SPARKAI_API_SECRET'
SPARKAI_API_KEY = 'YOUR_SPARKAI_API_KEY'
SPARKAI_DOMAIN = 'generalv3.5'

# 初始化Wikipedia API
wiki_wiki = wikipediaapi.Wikipedia('179739684@q.com', 'zh')

# 初始化讯飞星火API
spark = ChatSparkLLM(
    spark_api_url=SPARKAI_URL,
    spark_app_id=SPARKAI_APP_ID,
    spark_api_key=SPARKAI_API_KEY,
    spark_api_secret=SPARKAI_API_SECRET,
    spark_llm_domain=SPARKAI_DOMAIN,
    streaming=False,
)

def get_wikipedia_summary(query):
    page = wiki_wiki.page(query)
    if page.exists():
        return page.summary, page._attributes["fullurl"]
    else:
        return "I'm sorry, I couldn't find any information on that topic in Wikipedia.", ""

def get_spark_answer(question):
    messages = [ChatMessage(role="user", content=question)]
    handler = ChunkPrintHandler()
    try:
        response = spark.generate([messages], callbacks=[handler])
        # 尝试获取响应内容
        try:
            text = response.generations[0][0].text
            return text.strip()
        except AttributeError as e:
            return f"AttributeError: {e}"
    except SparkAIConnectionError as e:
        return f"SparkAIConnectionError: Error Code: {e.error_code}, Error: {e.message}"

def get_spark_keywords(question):
    keyword_prompt = f"请从以下文字中提取出一个最主要的，且有wiki词条的关键词,只要一个关键词语，单词，不需要多余的解释：\n{question}"
    return get_spark_answer(keyword_prompt)

def answer_question(question):
    try:
        # 先用讯飞星火大模型提取关键词
        spark_answer = get_spark_answer(question)
        if spark_answer.startswith("SparkAIConnectionError:"):
            return spark_answer
        spark_keywords = get_spark_keywords(spark_answer).strip('关键词是：')
        
        # 从Wikipedia获取相关信息
        wiki_summary, wiki_url = get_wikipedia_summary(spark_keywords)

        # 构建最终答案
        if wiki_url:
            final_answer = f"根据讯飞星火生成的关键词：{spark_keywords}\n\n根据Wikipedia的资料：\n{wiki_summary}\n\n来源: {wiki_url}"
        else:
            final_answer = f"根据讯飞星火生成的关键词：{spark_keywords}\n\n抱歉，我无法在Wikipedia中找到相关信息。"

        return final_answer
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Gradio界面
iface = gr.Interface(
    fn=answer_question,
    inputs="text",
    outputs="text",
    title="Wikipedia Q&A with Spark",
    description="Ask a question and get an answer from the Spark model with Wikipedia references."
)

if __name__ == "__main__":
    iface.launch(share=True)
