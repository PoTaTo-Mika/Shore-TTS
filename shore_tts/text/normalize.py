import re

# 因为我们的字典里只有"，。？！"这四个，所以要把其它的删掉
def normalize(text):
    # 先把英文的标点替换为中文的
    text = re.sub(r',', '，', text)  # 英文逗号替换为中文逗号
    text = re.sub(r'\.', '。', text)  # 英文句号替换为中文句号
    text = re.sub(r'\?', '？', text)  # 英文问号替换为中文问号
    text = re.sub(r'!', '！', text)   # 英文感叹号替换为中文感叹号
    
    # 然后去掉文本里其它的不必要字符，只保留中文字符和这四个标点
    text = re.sub(r'[^\u4e00-\u9fa5，。？！]', '', text)
    return text

if __name__ == "__main__":
    text = "这是,一个标点的测试?是的!<>:[]=+-_"
    print(normalize(text))