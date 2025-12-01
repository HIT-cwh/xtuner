import re
import csv
import sys

def extract_metrics_simple(log_path, output_path):
    """简化版本的提取函数"""
    pattern = re.compile(
        r'Epoch \d+ Step \d+/\d+ data_time: [\d.]+ lr: [\d.e+-]+ time: ([\d.]+) '
        r'text_tokens: ([\d.]+) total_consumed_tokens: ([\d.]+) total_loss: ([\d.]+), '
        r'reduced_llm_loss: ([\d.]+), reduced_balancing_loss: ([\d.]+), loss: ([\d.]+), '
        r'maxvio: ([\d.]+), efficient_attn_ratio: ([\d.]+) grad_norm: ([\d.]+) '
        r'max_memory: ([\d.]+) GB reserved_memory: ([\d.]+) GB tgs: ([\d.]+) e2e_tgs: ([\d.]+)'
    )
    
    headers = [
        'time', 'text_tokens', 'total_consumed_tokens', 'total_loss',
        'reduced_llm_loss', 'reduced_balancing_loss', 'loss', 'maxvio',
        'efficient_attn_ratio', 'grad_norm', 'max_memory_GB', 
        'reserved_memory_GB', 'tgs', 'e2e_tgs'
    ]
    
    with open(log_path, 'r') as f, open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        
        for line in f:
            match = pattern.search(line)
            if match:
                writer.writerow(match.groups())

# 使用方法
if __name__ == "__main__":
    log_file = "/mnt/shared-storage-user/suzhongling/xtuner_11_27/work_dirs/qwen30_our_agrs/20251201110459/logs/rank0.log"
    output_file = "/mnt/shared-storage-user/suzhongling/xtuner_11_27/my_data/qwen30_our_agrs.csv"
    extract_metrics_simple(log_file, output_file)
    print(f"提取完成！结果保存在: {output_file}")
