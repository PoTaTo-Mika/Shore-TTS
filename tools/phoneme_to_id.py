import os
import sys
import pypinyin
import json

# 设置根目录，确保能正确导入shore_tts模块
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)  # 获取项目根目录
sys.path.insert(0, root_dir)

from shore_tts.text.normalize import normalize

def extract_pinyin(text):
    text = normalize(text)
    return pypinyin.pinyin(text, style=pypinyin.Style.TONE3)

def pinyin_to_phoneme(text):

    # 读取lexicon.json
    with open('checkpoints/dict/lexicon.json', 'r', encoding='utf-8') as f:
        lexicon = json.load(f)
    # 读取symbol_index.json
    with open('checkpoints/dict/symbol_index.json', 'r', encoding='utf-8') as f:
        symbol_index = json.load(f)

    # 获取拼音
    pinyin_list = extract_pinyin(text)
    phoneme_sequence = []
    phoneme_indices = []
    
    for pinyin_item in pinyin_list:
        # 每个pinyin_item是一个列表，取第一个元素
        pinyin = pinyin_item[0]
        
        # 在lexicon中查找对应的音素
        if pinyin in lexicon:
            phonemes = lexicon[pinyin]
            phoneme_sequence.extend(phonemes)
            
            # 将音素转换为索引
            for phoneme in phonemes:
                if phoneme in symbol_index['symbol_to_index']:
                    phoneme_indices.append(symbol_index['symbol_to_index'][phoneme])
                else:
                    print(f"警告: 音素 '{phoneme}' 未在symbol_index中找到")
        else:
            print(f"警告: 拼音 '{pinyin}' 未在lexicon中找到")
    
    return {
        'phoneme_sequence': phoneme_sequence,
        'phoneme_indices': phoneme_indices,
        'pinyin_list': [item[0] for item in pinyin_list]
    }

def text_to_ids(text):
    result = pinyin_to_phoneme(text)
    return result['phoneme_indices']

def pinyin_to_ids(pinyin_list):
    """
    直接从拼音列表转换为音素ID列表
    Args:
        pinyin_list: 拼音列表，例如 ['ni3', 'hao3', 'ma']
    Returns:
        phoneme_ids: 音素ID列表
    """
    # 读取lexicon.json
    with open('checkpoints/dict/lexicon.json', 'r', encoding='utf-8') as f:
        lexicon = json.load(f)
    # 读取symbol_index.json
    with open('checkpoints/dict/symbol_index.json', 'r', encoding='utf-8') as f:
        symbol_index = json.load(f)

    phoneme_ids = []
    for pinyin in pinyin_list:
        # 在lexicon中查找对应的音素
        if pinyin in lexicon:
            phonemes = lexicon[pinyin]
            
            # 将音素转换为索引
            for phoneme in phonemes:
                if phoneme in symbol_index['symbol_to_index']:
                    phoneme_ids.append(symbol_index['symbol_to_index'][phoneme])
                else:
                    print(f"警告: 音素 '{phoneme}' 未在symbol_index中找到")
        else:
            print(f"警告: 拼音 '{pinyin}' 未在lexicon中找到")
    
    return phoneme_ids

if __name__ == "__main__":
    text = "罗德岛全舰正处于通常航行状态。博士，整理下航程信息吧"
    result = pinyin_to_phoneme(text)
    print(f"文本: {text}")
    print(f"拼音: {result['pinyin_list']}")
    print(f"音素序列: {result['phoneme_sequence']}")
    print(f"音素索引: {result['phoneme_indices']}")
    
    # 测试修复后的 phoneme_to_ids 函数
    pinyin_list = ['ni3', 'hao3', 'ma']
    print(f"\n测试拼音列表: {pinyin_list}")
    print(f"转换后的音素ID: {pinyin_to_ids(pinyin_list)}")
    
    # 验证与原来的结果一致性
    print(f"\n验证一致性:")
    print(f"原方法结果: {result['phoneme_indices']}")
    print(f"新方法结果: {pinyin_to_ids(result['pinyin_list'])}")
    print(f"结果一致: {result['phoneme_indices'] == pinyin_to_ids(result['pinyin_list'])}")
