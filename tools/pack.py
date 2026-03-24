import os
import tarfile
import tempfile
import argparse
from collections import defaultdict

class MultiVolumeReader:
    """
    用于连续读取多个分卷打包文件的文件流封装，
    能够在不占用额外硬盘空间的情况下流式读取所有分片。
    """
    def __init__(self, filenames):
        # 确保按 .000, .001 后缀顺序读取
        self.filenames = sorted(filenames)
        self.index = 0
        self.f = open(self.filenames[self.index], 'rb')
        
    def read(self, size=-1):
        if size < 0:
            res = [self.f.read()]
            self.f.close()
            self.index += 1
            for fn in self.filenames[self.index:]:
                with open(fn, 'rb') as f:
                    res.append(f.read())
            self.index = len(self.filenames)
            return b"".join(res)
            
        data = self.f.read(size)
        # 如果当前分卷已读完，且还有下一个分卷
        if len(data) < size and self.index < len(self.filenames) - 1:
            self.f.close()
            self.index += 1
            self.f = open(self.filenames[self.index], 'rb')
            data += self.read(size - len(data))
        return data
        
    def close(self):
        if self.f and not self.f.closed:
            self.f.close()

def process_dataset(input_folder, output_folder, shard_size=10000):
    print(f"开始扫描输入目录: {input_folder}")
    split_groups = defaultdict(list)
    
    # 1. 递归扫描并收集所有的分卷组
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if '.tar.' in file:
                # 寻找包含 .tar. 的文件，切分出基础名，例如 data.tar.000 -> data.tar
                base_name = file[:file.find('.tar.') + 4] 
                full_path = os.path.join(root, file)
                group_key = os.path.join(root, base_name)
                split_groups[group_key].append(full_path)
                
    if not split_groups:
        print("未找到任何符合条件的多分卷 tar 文件。")
        return

    # 2. 依次处理每个分卷组
    for group_key, parts in split_groups.items():
        parts = sorted(parts)
        base_tar_name = os.path.basename(group_key)
        dataset_name = base_tar_name.replace('.tar', '')
        
        # 在输出目录中重建与输入一样的目录树
        rel_dir = os.path.relpath(os.path.dirname(group_key), input_folder)
        out_dir = os.path.join(output_folder, rel_dir)
        os.makedirs(out_dir, exist_ok=True)
        
        print(f"\n[处理中] 归档组: {group_key} (共 {len(parts)} 个分片)")
        
        # 使用安全的临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            print("  -> 正在流式解压至内存临时区域...")
            try:
                fileobj = MultiVolumeReader(parts)
                with tarfile.open(fileobj=fileobj, mode='r|*') as tar:
                    # Python 3.12+ 安全解压过滤，兼容低版本
                    if hasattr(tarfile, 'data_filter'):
                        tar.extractall(path=temp_dir, filter='data')
                    else:
                        tar.extractall(path=temp_dir)
            except Exception as e:
                print(f"  [错误] 解压 {group_key} 失败: {e}")
                continue
            finally:
                fileobj.close()
                
            print("  -> 正在扫描和精准匹配 opus 与 txt 数据对...")
            samples = defaultdict(dict)
            for troot, _, tfiles in os.walk(temp_dir):
                for tf in tfiles:
                    if tf.endswith('.opus') or tf.endswith('.txt'):
                        full_path = os.path.join(troot, tf)
                        # 生成相对路径并替换路径分隔符，将数据完美展平，避免多个子文件夹下的同名文件冲突
                        rel_path = os.path.relpath(full_path, temp_dir)
                        flat_name = os.path.splitext(rel_path)[0].replace(os.sep, '_')
                        
                        ext = os.path.splitext(tf)[1] # '.opus' 或 '.txt'
                        samples[flat_name][ext] = full_path
                        
            # 过滤出成对的完整数据 (WebDataset严格要求配对)
            complete_pairs =[]
            for name, exts in samples.items():
                if '.opus' in exts and '.txt' in exts:
                    complete_pairs.append((name, exts['.opus'], exts['.txt']))
                    
            print(f"  -> 成功匹配 {len(complete_pairs)} 组完整数据。")
            if not complete_pairs:
                continue
                
            complete_pairs.sort() # 排序保证多次运行打包顺序的强一致性
            
            # 3. 按指定数量分片，重新打包为标准的 WebDataset Tar
            shard_idx = 0
            for i in range(0, len(complete_pairs), shard_size):
                chunk = complete_pairs[i:i + shard_size]
                shard_name = f"{dataset_name}_{shard_idx:06d}.tar"
                shard_path = os.path.join(out_dir, shard_name)
                
                print(f"  -> 正在写入分片: {shard_name} ({len(chunk)} 条数据)...")
                with tarfile.open(shard_path, "w") as out_tar:
                    for name, opus_path, txt_path in chunk:
                        # 压入新的tar时，展平结构 (如 sub_dir_audio1.opus)
                        out_tar.add(opus_path, arcname=f"{name}.opus")
                        out_tar.add(txt_path, arcname=f"{name}.txt")
                        
                shard_idx += 1
                
        print(f"完成 {dataset_name} 的全部数据转换！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将旧版的按卷分卷打包数据集转换为适用于训练的 WebDataset 分片数据集")
    parser.add_argument("-i", "--input", required=True, help="输入文件夹的路径")
    parser.add_argument("-o", "--output", required=True, help="输出文件夹的路径")
    parser.add_argument("-s", "--shard_size", type=int, default=10000, help="每个目标分片的样本数量 (默认: 10000)")
    
    args = parser.parse_args()
    
    # 执行处理逻辑
    process_dataset(args.input, args.output, args.shard_size)
    print("\n所有任务已处理完毕。")