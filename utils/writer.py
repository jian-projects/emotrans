# 写入文件、保存文件、打开文件
import os

from regex import P
# 删除文件
def removeFile(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)

## 0. csv 文件
"""
    csv_file = CsvFile('outs/result.csv', mode_w='a', delete=True)
    csv_file.write(['params','method']) # 写入内容是list, 元素是str
"""
import csv
class CsvFile(object):
    def __init__(self, file_path=None, encoding='utf-8', mode_w='a', mode_r='r', delete=True) -> None:
        self.mode_w = mode_w       # 写入方式 ['a':追加, 'w':重写]
        self.mode_r = mode_r       # 读取方式
        self.encoding = encoding   # 编码方式
        self.file_path = file_path # 文件路径
        if delete: removeFile(file_path) # 删除原有文件
        
    def write(self, content):
        assert '.csv' in self.file_path
        # 打开文件，并写入文件
        with open(self.file_path, mode=self.mode_w, encoding=self.encoding) as f:
            writer = csv.writer(f)    # 实例化写入器
            writer.writerow(content)  # 内容写入文件

    def read(self,):
        pass

## 1. json 文件
"""
    json_file = JsonFile('outs/result.json', mode_w='a', delete=True)
    json_file.write({'params': 1, 'result': 1})
"""
import json
class JsonFile(object):
    def __init__(self, file_path=None, encoding='utf-8', mode_w='a', mode_r='r', delete=True) -> None:
        self.mode_w = mode_w       # 写入方式 ['a':追加, 'w':重写]
        self.mode_r = mode_r       # 读取方式
        self.encoding = encoding   # 编码方式
        self.file_path = file_path # 文件路径
        if delete: removeFile(file_path) # 删除原有文件
    
    def write(self, content, space=False):
        # content 是list, 元素是个字典
        assert '.json' in self.file_path
        with open(self.file_path, mode=self.mode_w, encoding=self.encoding) as f:
            if space: f.write('\n')
            if isinstance(content, dict): content = {k: str(v) for k, v in content.items()}
            json_dict = json.dump(content, f)
            f.write('\n')
            
    def read(self,):
        assert '.json' in self.file_path
        with open(self.file_path, mode=self.mode_r, encoding=self.encoding) as f:
            json_dict = json.load(f)

        return json_dict


## 2. txt 文件
class TxtFile(object):
    def __init__(self, file_path=None, encoding='utf-8', mode_w='a', mode_r='r', delete=True) -> None:
        self.mode_w = mode_w       # 写入方式 ['a':追加, 'w':重写]
        self.mode_r = mode_r       # 读取方式
        self.encoding = encoding   # 编码方式
        self.file_path = file_path # 文件路径
        if delete: removeFile(file_path) # 删除原有文件
    
    def write(self, content):
        # content 是 list, 元素是 srt
        if not isinstance(content, list): content = [content]
        assert '.txt' in self.file_path
        with open(self.file_path, mode=self.mode_w, encoding=self.encoding) as f:
            for sent in content: f.write(sent.strip()+'\n')
                
            
    def read(self,):
        pass
        return json_dict