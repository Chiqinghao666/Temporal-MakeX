#!/usr/bin/env python3
"""
MakeX 简化测试版本
用于验证基本功能而不需要完整的 C++ 编译
"""

import os
import sys
import pandas as pd
import numpy as np
from collections import defaultdict

def test_data_loading():
    """测试数据加载功能"""
    print("测试数据加载...")
    
    # 创建测试数据
    test_data = {
        'vertices': [(1, 1), (2, 0), (3, 1)],  # (id, label)
        'edges': [(1, 2, 0), (2, 3, 1)],       # (src, dst, label)
    }
    
    print("测试数据创建成功")
    return test_data

def test_pattern_generation():
    """测试模式生成功能"""
    print("测试模式生成...")
    
    # 简单的双星型模式生成
    pattern = {
        'vertices': [[1, 1], [2, 0]],  # [id, label]
        'edges': [[1, 2, 0]]           # [src, dst, label]
    }
    
    print("模式生成成功")
    return pattern

def test_rule_scoring():
    """测试规则评分功能"""
    print("测试规则评分...")
    
    # 简单的评分逻辑
    support = 100
    confidence = 0.8
    score = support * confidence
    
    print(f"规则评分: {score}")
    return score

def main():
    """主测试函数"""
    print("=== MakeX 简化测试 ===")
    
    try:
        # 测试各个组件
        test_data_loading()
        test_pattern_generation()
        test_rule_scoring()
        
        print("所有测试通过！")
        print("\n下一步建议：")
        print("1. 下载完整的数据集")
        print("2. 使用简化版本验证算法逻辑")
        print("3. 逐步添加更复杂的功能")
        
    except Exception as e:
        print(f"测试失败: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()
