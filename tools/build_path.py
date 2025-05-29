import os
import sys
import pypinyin
from pathlib import Path

# 设置根目录，确保能正确导入模块
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)  # 获取项目根目录
sys.path.insert(0, root_dir)

try:
    from shore_tts.text.normalize import normalize
except ImportError:
    print("警告: 无法导入normalize模块，将使用原始文本")
    def normalize(text):
        return text

def extract_pinyin_from_text(text):
    """
    从文本中提取拼音
    Args:
        text: 输入文本
    Returns:
        pinyin_list: 拼音列表，例如 ['ni3', 'hao3']
    """
    try:
        # 尝试使用normalize函数
        normalized_text = normalize(text)
    except:
        # 如果normalize失败，使用原始文本
        normalized_text = text
    
    # 使用pypinyin提取拼音，带声调数字
    pinyin_result = pypinyin.pinyin(normalized_text, style=pypinyin.Style.TONE3)
    
    # 将结果展平为列表
    pinyin_list = [item[0] for item in pinyin_result]
    
    return pinyin_list

def build_path(directory_path, output_dir="data"):
    """
    遍历目录，找到.lab和.pt文件对，生成mel_list和pinyin_list
    
    Args:
        directory_path: 要遍历的目录路径
        output_dir: 输出文件的目录，默认为"data"
    """
    directory_path = Path(directory_path)
    output_dir = Path(output_dir)
    
    if not directory_path.exists():
        print(f"错误: 目录 {directory_path} 不存在")
        return
    
    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"开始遍历目录: {directory_path}")
    
    # 收集所有.lab和.pt文件
    lab_files = {}  # {文件名(不含扩展名): 完整路径}
    pt_files = {}   # {文件名(不含扩展名): 完整路径}
    
    # 递归遍历目录
    for file_path in directory_path.rglob("*"):
        if file_path.is_file():
            file_stem = file_path.stem  # 文件名不含扩展名
            
            if file_path.suffix.lower() == '.lab':
                lab_files[file_stem] = file_path
            elif file_path.suffix.lower() == '.pt':
                pt_files[file_stem] = file_path
    
    print(f"找到 {len(lab_files)} 个.lab文件")
    print(f"找到 {len(pt_files)} 个.pt文件")
    
    # 找到匹配的文件对
    matched_files = []
    for file_stem in lab_files:
        if file_stem in pt_files:
            matched_files.append((file_stem, lab_files[file_stem], pt_files[file_stem]))
    
    print(f"找到 {len(matched_files)} 对匹配的文件")
    
    if len(matched_files) == 0:
        print("警告: 没有找到匹配的.lab和.pt文件对")
        return
    
    # 按文件名排序，确保输出顺序一致
    matched_files.sort(key=lambda x: x[0])
    
    # 准备输出列表
    mel_list = []
    pinyin_list = []
    
    failed_count = 0
    
    for file_stem, lab_path, pt_path in matched_files:
        try:
            # 读取.lab文件内容
            with open(lab_path, 'r', encoding='utf-8') as f:
                text_content = f.read().strip()
            
            if not text_content:
                print(f"警告: {lab_path} 文件为空，跳过")
                failed_count += 1
                continue
            
            # 提取拼音
            pinyin_result = extract_pinyin_from_text(text_content)
            
            if not pinyin_result:
                print(f"警告: 无法从 {lab_path} 提取拼音，跳过")
                failed_count += 1
                continue
            
            # 添加到列表
            mel_list.append(str(pt_path))
            pinyin_list.append(' '.join(pinyin_result))
            
            print(f"处理成功: {file_stem}")
            print(f"  文本: {text_content}")
            print(f"  拼音: {' '.join(pinyin_result)}")
            print(f"  mel路径: {pt_path}")
            
        except Exception as e:
            print(f"处理文件 {file_stem} 时出错: {e}")
            failed_count += 1
            continue
    
    print(f"\n处理完成:")
    print(f"  成功处理: {len(mel_list)} 对文件")
    print(f"  失败: {failed_count} 对文件")
    
    if len(mel_list) == 0:
        print("错误: 没有成功处理任何文件")
        return
    
    # 写入mel_list.list
    mel_list_path = output_dir / "mel_list.list"
    with open(mel_list_path, 'w', encoding='utf-8') as f:
        for mel_path in mel_list:
            f.write(f"{mel_path}\n")
    
    # 写入pinyin_list.list
    pinyin_list_path = output_dir / "pinyin_list.list"
    with open(pinyin_list_path, 'w', encoding='utf-8') as f:
        for pinyin_line in pinyin_list:
            f.write(f"{pinyin_line}\n")
    
    print(f"\n输出文件:")
    print(f"  mel列表: {mel_list_path} ({len(mel_list)} 行)")
    print(f"  拼音列表: {pinyin_list_path} ({len(pinyin_list)} 行)")
    
    # 显示前几个示例
    print(f"\n前3个示例:")
    for i in range(min(3, len(mel_list))):
        print(f"  {i+1}. mel: {mel_list[i]}")
        print(f"     拼音: {pinyin_list[i]}")

def main():
    input_dir = 'data/mel'
    output_dir = 'data'
    
    build_path(input_dir, output_dir)

if __name__ == "__main__":
    main()
