"""
Byte Pair Encoding (BPE) 示例模块

该模块演示了 BPE 算法的基本实现，包括：
1. 从语料库中提取单词计数
2. 计算字符对的统计信息
"""

import collections
import regex as re


def get_word_counts(corpus: str) -> dict[tuple[str, ...], int]:
    """
    从给定语料库中提取单词并计算它们的出现次数。
    
    使用正则表达式模式将语料库分割成单词，然后将每个单词转换为字符元组，
    并统计每个字符序列的出现次数。
    
    Args:
        corpus (str): 输入的文本语料库
        
    Returns:
        dict[tuple[str, ...], int]: 字符元组到出现次数的映射字典
    """
    matches = re.finditer(PAT, corpus)
    return collections.Counter(tuple(match.group(0)) for match in matches)


def get_pair_stats(word_counts: dict[tuple[str, ...], int]) -> collections.Counter:
    """
    计算相邻字符对的统计信息。
    
    遍历所有单词及其计数，统计相邻字符对的出现次数。
    
    Args:
        word_counts (dict[tuple[str, ...], int]): 字符元组到出现次数的映射
        
    Returns:
        collections.Counter: 字符对到出现次数的计数器
    """
    pair_counts = collections.Counter()
    for word, count in word_counts.items():
        for i in range(len(word) - 1):
            pair = word[i : i + 2]
            pair_counts[pair] += count
    return pair_counts


if __name__ == "__main__":
    # 定义用于分词的正则表达式模式
    # 这个模式可以处理缩写、字母、数字、标点符号和空白字符
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    # 示例语料库，包含重复的单词用于演示 BPE 算法
    corpus = (
        "low low low low low "
        "lower lower widest widest widest "
        "newest newest newest newest newest newest"
    )
    
    # 获取单词计数并打印结果
    word_counts = get_word_counts(corpus)
    print(word_counts)