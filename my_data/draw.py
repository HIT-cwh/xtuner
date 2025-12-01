import pandas as pd
import matplotlib.pyplot as plt
import sys

def plot_reduced_llm_loss(csv1_path, csv2_path, num_rows=100):
    """
    绘制两个CSV文件中指定行数的reduced_llm_loss对比图
    
    Args:
        csv1_path: 第一个CSV文件路径
        csv2_path: 第二个CSV文件路径
        num_rows: 要绘制的行数，默认为100行
    """
    try:
        # 读取CSV文件
        df1 = pd.read_csv(csv1_path)
        df2 = pd.read_csv(csv2_path)
        
        # 检查必要的列是否存在
        required_columns = ['reduced_llm_loss']
        for df, name in zip([df1, df2], ['nccl', 'our']):
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"错误: {name}中缺少列: {missing_columns}")
                return
        
        # 检查是否有足够的数据行
        if len(df1) < num_rows:
            print(f"警告: nccl文件只有{len(df1)}行数据，少于请求的{num_rows}行")
            num_rows = len(df1)
        if len(df2) < num_rows:
            print(f"警告: our文件只有{len(df2)}行数据，少于请求的{num_rows}行")
            num_rows = min(num_rows, len(df2))
        
        # 取指定行数的数据
        df1_selected = df1.head(num_rows)
        df2_selected = df2.head(num_rows)
        
        # 创建x轴坐标
        x_axis = range(num_rows)
        
        # 创建图形
        plt.figure(figsize=(12, 8))
        
        # 绘制两个数据集
        plt.plot(x_axis, df1_selected['reduced_llm_loss'], 
                label='nccl', marker='o', markersize=3, linewidth=1.5)
        plt.plot(x_axis, df2_selected['reduced_llm_loss'], 
                label='our', marker='s', markersize=3, linewidth=1.5)
        
        # 设置图表属性
        plt.xlabel(f'Step (0-{num_rows-1})', fontsize=12)
        plt.ylabel('Reduced LLM Loss', fontsize=12)
        plt.title(f'Reduced LLM Loss Comparison (First {num_rows} Steps)', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        # 设置x轴范围
        plt.xlim(0, num_rows-1)
        
        # 显示图表
        plt.tight_layout()
        plt.savefig("/mnt/shared-storage-user/suzhongling/xtuner_11_27/my_data/loss.png")
        
        # 打印一些统计信息
        print(f"nccl - 数据点数: {len(df1_selected)}, Reduced LLM Loss范围: [{df1_selected['reduced_llm_loss'].min():.4f}, {df1_selected['reduced_llm_loss'].max():.4f}]")
        print(f"our - 数据点数: {len(df2_selected)}, Reduced LLM Loss范围: [{df2_selected['reduced_llm_loss'].min():.4f}, {df2_selected['reduced_llm_loss'].max():.4f}]")
        
    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
    except pd.errors.EmptyDataError:
        print("错误: CSV文件为空")
    except pd.errors.ParserError:
        print("错误: CSV文件格式不正确")
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    nccl = "/mnt/shared-storage-user/suzhongling/xtuner_11_27/my_data/qwen30_nccl_cwh.csv"
    our = "/mnt/shared-storage-user/suzhongling/xtuner_11_27/my_data/qwen30_our_agrs.csv"
    
    # 调用绘图函数，指定要绘制的行数
    # 例如：要绘制200行数据，只需将下面的100改为200
    plot_reduced_llm_loss(nccl, our, num_rows=150)
