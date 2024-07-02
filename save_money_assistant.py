from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

prompt_template = ChatPromptTemplate.from_messages(
    [
        ('system',"你被用于抑制用户的购买欲望。也就是说，当用户说想要买什么东西时，你需要提供合理的理由让用户不要购买。"),
        ('human',"我正在考虑购买一个{product}，但我想抑制购买欲望。你能帮我列出一些理由，让我思考一下我是否真的需要这个商品吗？")
    ]
)
# prompt_template.format(product="iPhone 13")

model = ChatOpenAI(
    model = 'moonshot-v1-32k',
    base_url = "https://api.moonshot.cn/v1",
    max_tokens=500,
    temperature=0.6
)

def output_parser(output: str):
    parser_model = ChatOpenAI(
        model = 'moonshot-v1-32k',
        temperature=0.8,
        openai_api_base = "https://api.moonshot.cn/v1"
    )
    
    message = "你需要将传入的文本进行改写，尽可能自然且符合人类对话习惯而不是采用类似键值对的形式回答。这是你需要改写的文本:'{text}'"
    return parser_model.invoke(message.format(text = output))

chain = prompt_template | model | output_parser

# using langchain pipeline
while True:
    product = input("你最近想买什么？")
    answer = chain.invoke(input = {'product' : product})
    print(answer.content)

